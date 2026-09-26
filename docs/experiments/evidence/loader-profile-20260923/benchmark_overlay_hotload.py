"""Eight read-only real overlay shards; constructor plus actual _load_one CPU diagnostic."""
from __future__ import annotations
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import shutil
import sys
import time

os.sched_setaffinity(0,{16,17})
signal.alarm(180)
resource.setrlimit(resource.RLIMIT_CPU,(120,120))
os.nice(19)
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['NUMEXPR_NUM_THREADS']='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
source=Path(sys.argv[1]);output=Path(sys.argv[2]);sys.path.insert(0,str(source))
import numpy as np
import numcodecs.blosc as blosc
import torch
blosc.set_nthreads(2);torch.set_num_threads(2)
from chess_anti_engine.replay import target_overlay as storage
from chess_anti_engine.replay import game_epoch as epoch
receipt=Path('/home/josh/projects/chess/scratchpad/bt4_joint20/factorial58_20260919/preparation/qualifications/B.json')
value=json.loads(receipt.read_text())
entries=value['shards'][:8]
paths=[Path(e['path']) for e in entries]
manifest=json.loads((paths[0]/storage.MANIFEST).read_text())
context=storage.BaseSeals([manifest['base_seal']])
if hasattr(context,'bind_roots'):
 context.bind_roots([paths[0].parent,context.root])
 context.bind_receipt(receipt,storage._receipt_stamp(receipt))
calls=[]
original=storage._open_target_manifest
def counted(path,manifest,*,seal=None):
 calls.append(str(path));return original(path,manifest,seal=seal)
storage._open_target_manifest=counted
stages={}
def guard():
 mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
 assert mem['MemAvailable'] >= 40*2**30, 'memory reserve'
 assert shutil.disk_usage(output.parent).free >= 150*2**30, 'disk reserve'
 assert not (output.parent/'STOP').exists(), 'STOP requested'
guard()
start=time.monotonic()
def mark(name,then):stages[name]=time.monotonic()-then
try:
 t=time.monotonic()
 for path,entry in zip(paths,entries):
  assert storage.overlay_content_sha256(path,seal=context)==entry['content_sha256']
 mark('selected_shard_identity',t)
 t=time.monotonic();records=epoch._scan_shards(paths,2,allow_target_overlay=True,overlay_seal=context);mark('scan_shards',t)
 def counter(arrays):
  return {'policy':float(np.asarray(arrays['policy_target']).sum(axis=1).size),'value':float(np.asarray(arrays['search_wdl']).sum(axis=1).size)}
 t=time.monotonic();records=epoch._attach_objective_mask_weights(records,counter,2,allow_target_overlay=True,overlay_seal=context);mark('objective_census',t)
 startup_calls=len(calls)
 t=time.monotonic();plan,ordered=epoch._plan_epoch(records,batch_size=128,seed=121,load_workers=1,max_working_set_bytes=4*2**30,mirror_augmentation=False);mark('plan',t)
 # Exercise the exact hot loader on genuinely scanned/planned records, using
 # this operation's validated context. No training constructor or GPU is run.
 loader=object.__new__(epoch.GameAwareEpochBuffer)
 loader._allow_target_overlay=True;loader._overlay_seal=context
 loader._input_planes=None;loader._objective_mask_counter=counter
 load_calls_before=len(calls);load_times=[];decoded_hashes={}
 for record in ordered:
  guard();t=time.monotonic();arrays=loader._load_one(record)
  load_times.append(time.monotonic()-t)
  decoded_hashes[str(record.path)]={name:{'shape':list(a.shape),'dtype':a.dtype.str,'sha256':hashlib.sha256(a.tobytes()).hexdigest()} for name,a in arrays.items()}
  del arrays
 stages['hot_load_one']=sum(load_times)
 hot_load_calls=len(calls)-load_calls_before
 # Ordered target bytes are banked alongside deterministic plan and counts.
 t=time.monotonic();target_digests={}
 for record in ordered:
  arrays,_=storage.overlay_proxies(record.path,('policy_target','search_wdl','game_id'),seal=context)
  target_digests[str(record.path)]={name:hashlib.sha256(np.asarray(array).tobytes()).hexdigest() for name,array in arrays.items()}
 mark('ordered_target_hashes',t)
 result={'decoded_array_hashes':decoded_hashes,'hot_load_seconds_per_shard':load_times,'hot_load_semantic_validations':hot_load_calls,'error':None,'source':str(source),'source_sha256':hashlib.sha256((source/'chess_anti_engine/replay/target_overlay.py').read_bytes()).hexdigest(),'shards':entries,'qualification_receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest(),'stages_seconds':stages,'startup_semantic_validations':startup_calls,'total_semantic_validations':len(calls),'plan':dataclasses.asdict(plan),'ordered_target_sha256':target_digests,'scope':'Eight real shards selected from published B receipt; constructor stages measured directly, plus actual per-record _load_one; not full-corpus qualification, training schedule or GPU performance.'}
except Exception as exc:
 result={'error':repr(exc),'source':str(source),'stages_seconds':stages,'semantic_validations':len(calls)}
result['wall_seconds']=time.monotonic()-start
r=resource.getrusage(resource.RUSAGE_SELF);result['cpu_seconds']=r.ru_utime+r.ru_stime
payload=json.dumps(result,indent=2,default=str)
assert len(payload.encode()) < 2**20, 'per-arm receipt limit1MiB'
assert not output.exists(), 'fresh output required'
output.write_text(payload);print(json.dumps({k:v for k,v in result.items() if k not in ['shards','plan','ordered_target_sha256','decoded_array_hashes']},indent=2))
