"""Tiny isolated operational-wrapper checks; never invokes real freeze/execute."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('increment', HERE/'run_common.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
checks = []
registered=m.read(m.REG)
m.validate_registration(registered)
bad=copy.deepcopy(registered);bad["derive_options"]["value_observation"]="phase0"
try:
    m.validate_registration(bad)
except ValueError:
    pass
else:
    raise AssertionError("changed value selector accepted")
checks.append("actual fixed registration accepted; changed value observation refused")


def reject(fn):
    try:
        fn()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError('invalid input accepted')


with tempfile.TemporaryDirectory(prefix='increment-fixture-') as folder:
    t = Path(folder)
    selected = [{'source_shard':'w00-00032.jsonl.zst','rows':100,'source_sha256':'0'*64}]
    selection = t/'selection.json'
    selection.write_text(json.dumps({'schema':1,'source_dir':str(t/'raw'),'shards':selected}))
    source = {'source_id':'run06_g10','source_dir':str(t/'raw'),'physical_rows':100,
              'selection':{'path':str(selection),'sha256':m.sha(selection)},
              'support_drop_ceiling':64,'missing_result_count_ceiling':2,
              'derived_output':str(t/'derived')}
    summary = {'source_selection':m.selection_proof(source),'max_policy_support_misses':64,
               'realized':{'rows_read':100,'rows_written':98,'rows_dropped_policy_support':0,
                           'policy_support_exclusions':[],'rows_dropped_no_result':2,'rows_dropped_envelope':0}}
    assert m.actual_exclusions(source,summary)==set()
    for key,value in [('rows_dropped_no_result',3),('rows_dropped_envelope',1),('rows_dropped_policy_support',65)]:
        bad=copy.deepcopy(summary);bad['realized'][key]=value
        reject(lambda bad=bad:m.actual_exclusions(source,bad))
    checks.append('actual missing-result ceiling accepted; excess missing/support/envelope counters refused')
    rank = {'source_selection':m.selection_proof(source),'raw_rows_read':100,'rows':98,
            'rows_dropped_no_result':2,'row_provenance':{'join':'source-qualified-physical-row-and-full-history-keys-v1'}}
    assert m.verify_complement(source,summary,rank,{'s':bytearray([1]*98+[0]*2)})==2
    reject(lambda:m.verify_complement(source,summary,rank,{'s':bytearray([1]*97+[0]*3)}))
    bad=copy.deepcopy(rank);bad['rows_dropped_no_result']=1
    reject(lambda:m.verify_complement(source,summary,bad,{'s':bytearray([1]*98+[0]*2)}))
    checks.append('independent rank count and injective-survivor cardinality detect omitted eligible row')
    raw=t/'raw';raw.mkdir();file=raw/selected[0]['source_shard'];file.write_bytes(b'original')
    stat=file.stat();metadata=t/'metadata.json'
    metadata.write_text(json.dumps([dict(zip(['device','inode','bytes','mtime_ns','ctime_ns'],m.identity(file)),source_path=str(file))]))
    source['source_metadata']={'path':str(metadata),'sha256':m.sha(metadata)}
    m.source_storage(source)
    (raw/'unselected-new-shard').write_bytes(b'allowed')
    m.source_storage(source)
    file.write_bytes(b'changed')
    reject(lambda:m.source_storage(source))
    checks.append('selected raw stat drift refused while unrelated appended file is allowed')
    # Capture actual worker command construction, replacing every stage with a no-op.
    fixture=t/'worker';fixture.mkdir();receipt=fixture/'receipts.jsonl'
    receipt.write_text(json.dumps({'sidecar':'s.zarr','onnx_sha256':'teacher','policy_output':'policy','providers':['CPU'],'remap_provenance':{}})+'\n')
    src={**source,'physical_rows':100,'derived_output':str(fixture/'derived'),
         'adapted_output':str(fixture/'bt4'),'rank_output':str(fixture/'rank'),
         'sidecar_dir':str(fixture/'rawbt4'),'closed_bt4_receipts':{'path':str(receipt)},'source_manifest':{}}
    reg={'sources':[src],'limits':{'adapter_index_cache_bytes':67108864,'rank_index_cache_bytes':67108864}}
    plan={'checkout':str(fixture),'python':sys.executable}
    original_read=m.read
    def fake_read(path):
        if Path(path)==m.REG:return reg
        if str(path).endswith('derive_targets_summary.json'):return {'realized':{'rows_written':98},'shards':[{}]}
        if str(path).endswith('.zattrs'):return {'onnx_path':'teacher.onnx'}
        return original_read(path)
    captured=[]
    with patch.object(m,'STATE',fixture),patch.object(m,'read',side_effect=fake_read),patch.object(m,'verify'),patch.object(m,'validate_registration'),patch.object(m,'source_storage'),patch.object(m,'identity',return_value=[1]),patch.object(m,'sha',return_value=source['selection']['sha256']),patch.object(m,'stage',side_effect=lambda n,a,p:captured.append((n,a))),patch.object(m,'guard'):
        m.worker(plan)
    assert [n.rsplit('.',1)[1] for n,a in captured]==['derive','snapshot','adapt','rank','qualify']
    for name,argv in captured:
        if name.endswith(('.derive','.rank')):
            assert argv[argv.index('--source-shards')+1]==str(selection)
            assert argv[argv.index('--limit')+1]=='100'
    derive=captured[0][1]
    assert derive[derive.index('--max-policy-support-misses')+1]=='64'
    assert derive[derive.index('--value-observation')+1]=='latest-phase'
    reject(lambda:m.command(plan,'lc0_control_train.py'))
    checks.append('actual worker builds only five allowed stages; derive/rank share exact selection and limits; training command refused')
    usage_calls=[]
    with patch.object(m,'STATE',t),patch.object(m,'_last_size_check',float('-inf')),patch.object(m,'usage',side_effect=lambda:usage_calls.append(1) or 0),patch.object(m.shutil,'disk_usage',return_value=type('Disk',(),{'free':200*1024**3})()),patch.object(m.time,'monotonic',side_effect=[0,5,61]):
        m.guard();m.guard();m.guard()
    assert len(usage_calls)==2
    checks.append('output-size walk sampled60s, not every5s guard')
    # Real owned grandchild ignores TERM; unrelated fixture process must survive.
    pidfile=t/'grandchild.pid'
    childcode="import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)']); Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
    owned=subprocess.Popen([sys.executable,'-c',childcode,str(pidfile)],start_new_session=True)
    unrelated=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
    try:
        deadline=time.monotonic()+3
        while not pidfile.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert pidfile.exists();grandchild=int(pidfile.read_text());time.sleep(.05)
        m.stop_owned_group(owned,grace=.1)
        assert owned.poll() is not None and unrelated.poll() is None
        deadline=time.monotonic()+2
        while Path(f'/proc/{grandchild}/stat').exists() and time.monotonic()<deadline:
            if Path(f'/proc/{grandchild}/stat').read_text().rsplit(')',1)[1].split()[0]=='Z':break
            time.sleep(.02)
        if Path(f'/proc/{grandchild}/stat').exists():assert Path(f'/proc/{grandchild}/stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
    finally:
        m.stop_owned_group(owned,grace=.1)
        unrelated.terminate();unrelated.wait()
    checks.append('owned group including TERM-ignoring grandchild reaped; unrelated fixture process preserved')

result={'status':'PASS_ISOLATED_WRAPPER_FIXTURES','checks':checks,'runner_sha256':m.sha(HERE/'run_common.py'),'scope':'Temporary data and owned CPU fixture subprocesses only; no operational corpus, freeze, launch or GPU.'}
with (HERE/'wrapper_qualification_v2.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
