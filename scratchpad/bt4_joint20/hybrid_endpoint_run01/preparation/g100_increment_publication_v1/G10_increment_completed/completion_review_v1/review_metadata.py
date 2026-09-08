"""Review completed receipts, small lineage metadata and NPZ byte pins only."""
from pathlib import Path
import datetime
import hashlib
import json
import time

STATE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
PINS = {}

def sha(path):
    with Path(path).open('rb') as f:
        h=hashlib.sha256()
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):
    path=Path(path);PINS[str(path)]=sha(path);return json.loads(path.read_text())

def same(a,b):return json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)

def identity(p):
    s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]

started=time.monotonic()
plan=read(STATE/'launch.json');complete=read(STATE/'completed.json');worker=read(STATE/'worker_complete.json')
assert complete['status']==worker['status']=='complete' and complete['exit_code']==0
assert complete['plan_sha256']==sha(STATE/'launch.json')=='87a33a55678ff0c02daa61bfe7aaab0acb82dd019753247a26c8a1323d9aa752'
assert 0<complete['end_unix']-complete['start_unix']<=7200
assert worker['end_unix']<=complete['end_unix'] and not (STATE/'failed.json').exists()
assert sha(STATE/'run_common.py')=='196464104f2da3cace69032212098356d497296575e0f0a31dc5c3f8dfe0ee86'
PINS[str(STATE/'run_common.py')]=sha(STATE/'run_common.py')
rows=[];stage_results={};lineage=[];raw_count=0
for name in ['run06_g10','run07_g10_companion4']:
    lane=STATE/name;q=read(lane/'common_input_qualification.json');sel=read(lane/'source_shards.json')
    assert q['status']=='complete';requested=sel['shards'];got=q['source_selection']
    assert {k:v for k,v in got.items() if k not in ['path','sha256','order']}==sel
    assert got['path']==str(lane/'source_shards.json') and got['sha256']==sha(lane/'source_shards.json')
    assert len(requested)==32 and [e['source_shard'] for e in requested]==[f'w00-{k:05d}.jsonl.zst' for k in range(32,64)]
    for path,h in q['summary_pins'].items():assert sha(Path(path))==h
    d=read(lane/'derived/derive_targets_summary.json');b=read(lane/'bt4/bt4_policy_sidecar_summary.json');r=read(lane/'rank/sf_d9_rank_sidecar_summary.json')
    assert same(q['derive_realized'],d['realized']);assert d['source_selection']==r['source_selection']==got
    real=d['realized'];n=real['rows_written'];missing=real['rows_dropped_no_result'];support=real['rows_dropped_policy_support'];physical=sum(e['rows'] for e in requested)
    assert physical==q['physical_rows']==real['rows_read']==r['raw_rows_read']
    assert n==q['rows']==b['rows']==r['rows']==physical-missing-support
    assert real['rows_dropped_envelope']==0 and support<=64 and missing<=int(.02*physical)
    assert q['independent_rank_missing_result_rows']==r['rows_dropped_no_result']==missing
    assert q['verified_support_exclusion_rows']==r['rows_dropped_policy_support']==r['row_provenance']['rows_dropped_policy_support']==support
    assert q['omitted_rows']==missing+support
    assert sum(q['per_raw_shard_survivors'].values())==n
    assert set(q['per_raw_shard_survivors'])=={e['source_shard'] for e in requested}
    assert all(0<=q['per_raw_shard_survivors'][e['source_shard']]<=e['rows'] for e in requested)
    assert r['row_provenance']['join']=='source-qualified-physical-row-and-full-history-keys-v1'
    assert r['top_k']==3 and r['row_provenance']['raw_shards_read_once']==32
    assert r['row_provenance']['policy_observation']=='phase0' and r['row_provenance']['value_observation']=='latest-phase'
    assert r['source_derive_summary_sha256']==sha(lane/'derived/derive_targets_summary.json')
    ledger=lane/'derived/policy_support_misses.jsonl';PINS[str(ledger)]=sha(ledger)
    refs=[json.loads(x) for x in ledger.read_text().splitlines()];assert same(refs,real['policy_support_exclusions']) and len(refs)==support
    assert sha(ledger)==r['row_provenance']['policy_support_exclusions_sha256']
    assert len({(e['source_dir'],e['source_shard'],e['source_row']) for e in refs})==support
    universe={e['source_shard']:e['rows'] for e in requested}
    for e in refs:
        assert e['source_dir']==sel['source_dir'] and 0<=e['source_row']<universe[e['source_shard']]
        assert e['reason']=='selected_phase0_policy_support' and e['full_history_input_key_verified'] is True and e['policy_depth']==9
    ds={e['path']:e for e in d['shards']};bs={e['path']:e for e in b['adapter']['written_shards']};rs={e['path']:e for e in r['outputs']}
    assert ds.keys()==bs.keys()==rs.keys() and len(ds)==r['shards'] and sum(e['rows'] for e in ds.values())==n
    for kind in ['derived','bt4','rank']:assert {p.name for p in (lane/kind).glob('shard_*.zarr')}==set(ds)
    for shard,e in ds.items():
        da=read(lane/'derived'/shard/'.zattrs');ba=read(lane/'bt4'/shard/'.zattrs');ra=read(lane/'rank'/shard/'.zattrs')
        npz=lane/'derived'/shard/'row_provenance.npz';nh=sha(npz);PINS[str(npz)]=nh
        assert nh==e['row_provenance']['sha256']==da['derive_row_provenance']['sha256']==bs[shard]['row_provenance_sha256']==ba['row_provenance_sha256']
        assert ba['bt4_policy_sha256']==bs[shard]['bt4_policy_sha256'] and ra['payload_sha256']==rs[shard]['payload_sha256']
        assert ba['source_derive_summary_sha256']==ra['source_derive_summary_sha256']==sha(lane/'derived/derive_targets_summary.json')
        assert ba['positions']==ra['source_rows']==bs[shard]['rows']==rs[shard]['rows']==e['rows']
        lineage.append({'source':name,'shard':shard,'rows':e['rows'],'row_provenance_sha256':nh,'bt4_payload_sha256':ba['bt4_policy_sha256'],'rank_payload_sha256':ra['payload_sha256']})
    for e in requested:
        p=Path(sel['source_dir'])/e['source_shard'];key=str(p)
        assert identity(p)==worker['raw_storage_identities'][key]==r['row_provenance']['raw_source_metadata'][key];raw_count+=1
    previous=complete['start_unix']
    for stage in ['derive','snapshot','adapt','rank','qualify']:
        st=read(STATE/f'{name}.{stage}.completed.json');assert st['exit_code']==st['resources']['exit_code']==0
        assert previous<=st['start_unix']<st['end_unix']<=worker['end_unix'];previous=st['end_unix']
        argv=st['argv']
        if stage in ['derive','rank']:assert argv[argv.index('--source-shards')+1]==str(lane/'source_shards.json') and int(argv[argv.index('--limit')+1])==physical
        stage_results[name+'.'+stage]=st['resources']
    rows.append({'source':name,'physical_rows':physical,'rows':n,'derived_shards':len(ds),'missing_results':missing,'support_exclusions':support,'envelope_drops':0,'sequence_sha256':q['source_qualified_input_sequence_sha256']})
assert raw_count==len(worker['raw_storage_identities'])==64
for p,h in PINS.items():assert sha(Path(p))==h
result={'status':'PASS_COMPLETED_INCREMENT_METADATA_REVIEW','reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_review_seconds':time.monotonic()-started,'sources':rows,'total':{'physical_rows':sum(e['physical_rows'] for e in rows),'rows':sum(e['rows'] for e in rows),'derived_shards':sum(e['derived_shards'] for e in rows),'missing_results':sum(e['missing_results'] for e in rows),'support_exclusions':sum(e['support_exclusions'] for e in rows),'wall_seconds':complete['end_unix']-complete['start_unix'],'ended_utc':datetime.datetime.fromtimestamp(complete['end_unix'],datetime.timezone.utc).isoformat()},'stage_resources':stage_results,'lineage':lineage,'metadata_sha256':PINS,'review_script_sha256':sha(Path(__file__)),'checks':['Completed supervisor/worker and ten zero-exit stages under120-minute cap','Exact registered64-source-shard selection, current raw storage stat identities, matching source-qualified omission and survivor counts','Six final summary hashes,65 derived/BT4/rank attr lineage joins and65 NPZ byte digests stable','Per-shard survivor counts sum to exact eligible complement, consistent with previously executed all-row qualifier and independent rank eligibility'],'limits':['No raw/feature/policy/rank payload replay, rederivation or teacher inference. NPZ lineage files were byte-hashed but not decoded/replayed.','Full legal-distribution and raw eligibility checks rely on the successfully executed pinned consumers/qualifier, not a second validation scan.','Reviewer authored earlier pilot plumbing; this is an independent completed-receipt/accounting review, not a new independent audit of every inherited qualifier implementation.','No C400 outcomes inspected; no training schedule or strength conclusion.']}
(OUT/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'total':result['total'],'sources':rows,'review_sha256':sha(OUT/'review.json'),'elapsed_review_seconds':result['elapsed_review_seconds']}))
