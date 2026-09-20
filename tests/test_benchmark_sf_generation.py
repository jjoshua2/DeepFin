"""Closed-game accounting and process ownership for CPU throughput screens."""
import gzip,json,os,signal
from pathlib import Path
import pytest
from scripts import benchmark_sf_generation as tool

def test_tracker_keeps_escaped_child_and_does_not_signal_reused_pid(monkeypatch):
    first={10:{'parent':1,'start':100,'ticks':2,'state':'S'},20:{'parent':10,'start':200,'ticks':5,'state':'S'},30:{'parent':20,'start':300,'ticks':9,'state':'S'}}
    monkeypatch.setattr(tool,'snapshot',lambda:first)
    owned=tool.OwnedProcesses(10);owned.sample()
    # Engine30 has its own session and outlives a worker; ancestry was retained.
    second={30:{'parent':1,'start':300,'ticks':15,'state':'S'},20:{'parent':1,'start':999,'ticks':0,'state':'S'}}
    monkeypatch.setattr(tool,'snapshot',lambda:second);assert owned.sample()==22/os.sysconf('SC_CLK_TCK')
    signalled=[];monkeypatch.setattr(tool.os,'kill',lambda pid,sig:signalled.append((pid,sig)))
    owned.signal(signal.SIGTERM);assert signalled==[(30,signal.SIGTERM)]

@pytest.mark.parametrize('depth',[8,9])
def test_closed_rows_are_checked_for_actual_policy_depth_and_results(tmp_path,depth):
    from tests.test_sf_policy_rewrite import raw_row
    rows=[raw_row(game_id=i) for i in range(3)]
    for row in rows:
        row['phases'][0]['per_depth'][0]['depth']=depth
    rows[1]['result']=None
    rows[2]['phases'][0]['per_depth'][0]['lines'][0][1]='a1a8'
    config=rows[0]['run']['config_sha256'];shard=tmp_path/'w00-00000.jsonl.gz'
    with gzip.open(shard,'wt') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    phases=[{'width':'all','depth':8}] if depth==8 else [{'width':'all','depth':9},{'width':8,'depth':10},{'width':4,'depth':12}]
    (tmp_path/'manifest.json').write_text(json.dumps({'config_sha256':config,'staircase_parsed':phases,'staircase_gate':{'policy':'fixed' if depth==8 else 'g10'}}))
    (tmp_path/'w00.progress.jsonl').write_text(json.dumps({'path':str(shard),'rows':3})+'\n'+json.dumps({'path':None,'rows':0,'games':[4]})+'\n'+ '{"torn":')
    (tmp_path/'unlisted.jsonl.gz').write_bytes(b'not a valid shard')
    result=tool.closed_readout(tmp_path,depth)
    assert result['banked_rows']==3 and result['no_result_rows']==1 and result['eligible_rows']==1 and result['invalid_rows']==1

def test_tracker_does_not_adopt_reused_root_pid(monkeypatch):
    first={10:{'parent':1,'start':100,'ticks':1,'state':'S'}}
    monkeypatch.setattr(tool,'snapshot',lambda:first);owned=tool.OwnedProcesses(10);owned.sample()
    replacement={10:{'parent':1,'start':999,'ticks':80,'state':'S'}}
    monkeypatch.setattr(tool,'snapshot',lambda:replacement);assert owned.sample()==1/os.sysconf('SC_CLK_TCK')
    monkeypatch.setattr(tool.os,'kill',lambda *a:pytest.fail('reused root signalled'));owned.signal(signal.SIGTERM)


def test_adopts_engine_that_escapes_before_first_sample(tmp_path):
    import subprocess,sys,time
    baseline=tool.child_baseline()
    pidfile=tmp_path/'pid'
    code="import subprocess,os,pathlib; p=subprocess.Popen(['sleep','60'],start_new_session=True); pathlib.Path(%r).write_text(str(p.pid))" % str(pidfile)
    worker=subprocess.Popen([sys.executable,'-c',code],start_new_session=True)
    owned=tool.OwnedProcesses(worker.pid,baseline)
    worker.wait(timeout=5)
    engine=int(pidfile.read_text())
    try:
        owned.sample()
        assert engine in owned.known
        owned.stop(worker)
        assert engine not in tool.snapshot()
    finally:
        try:os.kill(engine,signal.SIGKILL)
        except ProcessLookupError:pass
        try:os.waitpid(engine,0)
        except ChildProcessError:pass

def test_readout_checks_budget_before_opening_inputs(tmp_path):
    def stop():raise InterruptedError('STOP')
    with pytest.raises(InterruptedError,match='STOP'):
        tool.closed_readout(tmp_path,8,stop)


def test_registered_python_direct_script_readout_routes_local_package(tmp_path):
    """Real process with no PYTHONPATH, unlike pytest's injected repo imports."""
    import subprocess
    from tests.test_sf_policy_rewrite import raw_row
    row=raw_row(game_id=1)
    row['phases'][0]['per_depth'][0]['depth']=8
    bank=tmp_path/'bank';bank.mkdir()
    shard=bank/'w00-00000.jsonl.gz'
    with gzip.open(shard,'wt') as stream:stream.write(json.dumps(row)+'\n')
    (bank/'manifest.json').write_text(json.dumps({'config_sha256':row['run']['config_sha256'],'staircase_parsed':[{'width':'all','depth':8}],'staircase_gate':{'policy':'fixed'}}))
    (bank/'w00.progress.jsonl').write_text(json.dumps({'path':str(shard),'rows':1})+'\n')
    runtime=Path(tool.__file__).resolve().parents[1]
    # Simulate the registered absolute script entry point from a foreign cwd;
    # the host has a conflicting user-site package named scripts.
    program="import runpy,json; m=runpy.run_path(%r,run_name='readout_smoke'); print(json.dumps(m['closed_readout'](__import__('pathlib').Path(%r),8)))" % (str(runtime/'scripts/benchmark_sf_generation.py'),str(bank))
    env={**os.environ,'PYTHONPATH':'','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    result=subprocess.run(['/usr/bin/python3','-c',program],cwd=tmp_path,env=env,capture_output=True,text=True,timeout=60,check=True)
    counts=json.loads(result.stdout.strip().splitlines()[-1])
    assert counts['eligible_rows']==1 and counts['invalid_rows']==0
