import os,sys,time,json,pathlib,cProfile,pstats,io,hashlib,resource
import numpy as np,zarr
sys.path.insert(0,'/home/josh/projects/chess-worktrees/exact-gather-20260922')
from chess_anti_engine.replay.game_epoch import _slice_epoch_arrays
from chess_anti_engine.replay.disk_buffer import _concat_sparse_batches
from chess_anti_engine.replay.shard import _SHARD_FIELDS
start=time.monotonic();r=pathlib.Path('/home/josh/chess-artifacts/labels/factorial58_20260919/outputs/cohort00/D');arrays=[];pins=[]
for i in range(4):
 p=r/f'shard_{i:06d}.zarr';m=json.loads((p/'target_overlay.json').read_text());base=zarr.open_group(m['base'],mode='r');overlay=zarr.open_group(str(p),mode='r')
 a={k:np.asarray(base[k][:256] if base[k].shape else base[k][...]) for k in base.array_keys() if k in _SHARD_FIELDS}
 for k in ['policy_target','search_wdl']:a[k]=np.asarray(overlay[k][:256])
 arrays.append(a);pins.append({'overlay':str(p),'manifest_sha256':hashlib.sha256((p/'target_overlay.json').read_bytes()).hexdigest()})
ii=[np.arange(127,-1,-1,dtype=np.int64) for _ in arrays]
def old():return _concat_sparse_batches([_slice_epoch_arrays(a,i) for a,i in zip(arrays,ii)])
def new():
 # Prototype only: identical schemas/dtypes, valid nonnegative indices in this fixed screen.
 out=_concat_sparse_batches([{k:v[:0] if v.ndim else v for k,v in a.items()} for a in arrays])
 for k,v in list(out.items()):
  if v.ndim:
   target=np.empty((512,*v.shape[1:]),dtype=v.dtype);offset=0
   for a,i in zip(arrays,ii):np.take(a[k],i,axis=0,out=target[offset:offset+len(i)],mode='clip');offset+=len(i)
   out[k]=target
 return out
before=old();after=new();assert before.keys()==after.keys()
for k in before:assert before[k].dtype==after[k].dtype and np.array_equal(before[k],after[k],equal_nan=True),(k,before[k].dtype,after[k].dtype)
del before,after
runs=[]
for name in ['old','new','new','old','old','new']:
 fn=old if name=='old' else new
 t=time.monotonic()
 for _ in range(100):value=fn()
 runs.append({'variant':name,'iterations':100,'seconds':time.monotonic()-t})
profile=cProfile.Profile();profile.enable()
for _ in range(100):old()
profile.disable();s=io.StringIO();pstats.Stats(profile,stream=s).sort_stats('cumulative').print_stats(15)
result={'status':'FIXED_REAL_VALUES_PROTOTYPE_PARITY_PASS','rows_loaded_per_shard':256,'shards':pins,'batch_rows':512,'source_arrays_bytes':sum(v.nbytes for a in arrays for v in a.values()),'runs':runs,'profile':s.getvalue(),'wall_seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limitations':'Partial4shard tensors, repeated fixed gather only; not actual epoch schedule/trainer timing; homogeneous prototype with no mixed/default/error handling.'}
pathlib.Path('/tmp/gather-cpu-screen-20260922.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['runs','wall_seconds','peak_rss_kib']}));print(s.getvalue())
