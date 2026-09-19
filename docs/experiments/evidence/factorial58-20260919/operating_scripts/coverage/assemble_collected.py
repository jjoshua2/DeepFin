"""Assemble manifests from completed exact grouped collections; no inference.

Use only after all driver manifests complete. Collector already read back full
payloads; this assembly validates completed receipts and immutable metadata.
Training target admission remains the downstream producer's responsibility.
"""
import hashlib,json,os,tempfile
from pathlib import Path
B=Path(__file__).resolve().parents[1]/'collection'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def publish(path, value):
 if path.exists():
  if read(path)!=value: raise ValueError('existing assembly differs: '+str(path))
  return
 fd, temporary = tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
 try:
  with os.fdopen(fd,'w') as f:
   json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  try: os.link(temporary,path)
  except FileExistsError:
   if read(path)!=value: raise ValueError('concurrent assembly differs: '+str(path))
  directory_fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
  try: os.fsync(directory_fd)
  finally: os.close(directory_fd)
 finally: os.unlink(temporary)
index=read(B/'blocks.prepared.json'); manifests=[]
for block in index['blocks']:
 d=Path(block['driver_plan']).parent;driver=read(block['driver_plan']);terminal=read(d/'driver/manifest.json')
 assert terminal['status']=='COMPLETE' and terminal['plan_sha256']==sha(block['driver_plan'])
 receipts={x['id']:x for x in terminal['completed_chunks']}; entries=[]
 assert set(receipts)=={x['id'] for x in driver['chunks']}
 for c in driver['chunks']:
  receipt=receipts[c['id']]; assert sha(receipt['completion_path'])==receipt['completion_sha256']
  p=read(d/c['id']/'plan.json');complete=read(d/c['id']/'completed.json')
  assert complete['status']=='COMPLETE_CHUNK_PENDING_INDEPENDENT_REVIEW' and complete['plan_sha256']==sha(d/c['id']/'plan.json')
  for spec in p['selection']['shards']:
   path=Path(c['output_directory'])/spec['path']; attrs=read(path/'.zattrs');binding=attrs['binding']
   assert attrs['complete'] and binding['source']==block['source'] and binding['shard']==spec['path'] and binding['rows']==spec['rows']
   assert binding['summary_sha256']==p['selection']['summary_sha256'] and binding['audited_source_admission']['qualification']==block['source_manifest']
   assert binding['backend']['outputs']==['policy','value','value2']
   for name in ['policy_logits','value_logits','value2_logits','game_id','ply_index','row_index','tpg_feed_sha256','legal_indices','legal_offsets']:
    assert (path/name/'.zarray').is_file() and name in attrs['array_sha256']
   entries.append({'shard':spec['path'],'ceres':str(path),'ceres_binding':binding})
 assert len(entries)==block['shards'] and sum(e['ceres_binding']['rows'] for e in entries)==block['rows']
 output=d/'ceres_policy_manifest.json';result={'schema':1,'source':block['source'],'source_summary_sha256':sha(Path(block['source'])/'derive_targets_summary.json'),'entries':entries}
 publish(output,result)
 manifests.append({'path':str(output),'sha256':sha(output),'rows':block['rows'],'shards':block['shards']})
publish(B/'manifests.complete.json',{'status':'COMPLETE_COLLECTED_MANIFESTS_NOT_TRAINING_ADMISSION','manifests':manifests})
