from pathlib import Path
import json,hashlib,datetime,stat
p=Path('/home/josh/projects/chess/scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/G10_common_worker01_readiness_v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
d=read(p/'metadata_snapshot.json');count=0
for f in d['files']:assert sha(Path(f['path']))==f['sha256'];count+=1
for s in d['sources']:
 assert sha(Path(s['source_manifest']['path']))==s['source_manifest']['sha256']
 for m in read(Path(s['source_metadata']['path'])):
  st=Path(m['source_path']).lstat();assert stat.S_ISREG(st.st_mode) and st.st_nlink==1
  for k,a in [('device','st_dev'),('inode','st_ino'),('bytes','st_size'),('mtime_ns','st_mtime_ns'),('ctime_ns','st_ctime_ns')]:assert m[k]==getattr(st,a)
  assert sha(Path(m['sidecar_path'])/'.zattrs')==m['sidecar_attrs_snapshot']['sha256']
 for k in ['derived_output','adapted_output','rank_output']:assert not Path(s[k]).exists()
out={'status':'PASS_READINESS_ONLY','metadata_sha256':sha(p/'metadata_snapshot.json'),'prepared_sha256':sha(p/'prepared.json'),'metadata_files_checked':count,'selected_source_stats_and_attrs_rechecked':256,'raw_payload_reads':0,'policy_chunk_reads':0,'commands':['nice -n 19 ionice -c 3 taskset -c 6,7 python3 /tmp/prepare_w01_readiness_actual.py','nice -n 19 ionice -c 3 taskset -c 6,7 python3 /tmp/validate_w01_readiness.py'],'collector':{'path':str(p/'prepare_metadata.py'),'sha256':sha(p/'prepare_metadata.py')},'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'notes':['Initial taskset compact option refused before execution; fixed spacing.','Initial metadata collector compared entire stat including atime; first metadata read updated atime and assertion refused. Incomplete state preserved at G10_common_worker01_readiness_incomplete_atime. Successful collector explicitly checks dev/inode/size/mtime/ctime, never raw payloads.','Source objects retain historical support_drop_ceiling64 and missing_result_fraction_ceiling0.02 as reference defaults only, not a registered choice. Parent must finalize omission/resource/runtime contracts.','Four-hour cap, two two-core lanes, two numeric threads,8GiB sampled output and150GiB reserve are intended future bounds, not launched or qualified concurrency.','No registration, launch manifest, pipeline outputs or runtime changes created.']}
(p/'readiness_validation.json').write_text(json.dumps(out,indent=2)+'\n');(p/'validate_metadata.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({'receipt':str(p/'readiness_validation.json'),'sha256':sha(p/'readiness_validation.json'),'rows':d['physical_rows'],'files':count},indent=2))
