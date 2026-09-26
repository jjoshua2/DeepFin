#!/usr/bin/env python3
"""Disposable process/metadata checks only; never opens corpus or archive payloads."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

STATE = Path(__file__).resolve().parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False

def until(test, seconds=5):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if test():
            return
        time.sleep(.05)
    raise AssertionError('bounded condition did not arrive')

def main():
    started=time.time(); checks=[]
    h=load('archive_helper',STATE/'archive_G20T1.py')
    op=load('archive_operator',STATE/'run_registered.py')
    plan=json.loads((STATE/'plan.json').read_text())
    for pin in plan['input_pins']:
        assert hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()==pin['sha256']
    command=op.build_command(plan,time.time()+14400)
    assert command[command.index('/usr/bin/flock')+1:command.index('/usr/bin/flock')+4]==['--exclusive','--nonblock','--no-fork']
    assert command[command.index('/usr/bin/taskset')+1:command.index('/usr/bin/taskset')+3]==['-c','0,1']
    assert command[command.index('/usr/bin/ionice')+1:command.index('/usr/bin/ionice')+3]==['-c','3']
    assert command[command.index('--pool')+1]=='qtemp_0.0005_hist_20m_bt4_global_G20T1'
    for deadline in [float('nan'),float('inf'),time.time()+29,time.time()+14410]:
        try: op.build_command(plan,deadline)
        except ValueError: pass
        else: raise AssertionError('bad deadline admitted')
    checks.append('exact selected command, resources, shared lock and invalid deadlines')
    before=ast.parse((STATE/'archive_pools.original.py').read_text())
    after=ast.parse((STATE/'archive_G20T1.py').read_text())
    for name in ['stamp','assert_stat','inventory','check_source','verify_tar','walk','mount_check']:
        a=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name==name)
        if name == 'mount_check':
            for node in ast.walk(b):
                if isinstance(node, ast.Constant) and node.value == '/usr/bin/findmnt':
                    node.value = 'findmnt'
        assert ast.dump(a)==ast.dump(b),name
    checks.append('original source inventory, stability, tar member verification and mount algorithms unchanged')
    with tempfile.TemporaryDirectory(prefix='g20-archive-fixture-') as temp:
        d=Path(temp); h.ROOT=d
        (d/'STOP').touch()
        try: h.resources()
        except RuntimeError as e: assert 'STOP' in str(e)
        else: raise AssertionError('STOP ignored')
        (d/'STOP').unlink()
        checks.append('STOP refuses before source/destination access')
        sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(50)'])
        family=d/'family.py'
        family.write_text("import os,signal,time\nfrom pathlib import Path\nimport sys\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nchild=os.fork()\nif child==0:\n while True: time.sleep(.1)\nPath(sys.argv[1]).write_text(str(os.getpid())+' '+str(child))\nwhile True: time.sleep(.1)\n")
        pids=d/'pids'
        try:
            h.DEADLINE=time.time()+100
            h.CHILD=h.spawn([sys.executable,str(family),str(pids)])
            until(pids.exists)
            owned=list(map(int,pids.read_text().split()))
            h.cleanup_child(grace=.15)
            until(lambda: all(not alive(p) for p in owned))
            assert sentinel.poll() is None
            checks.append('TERM-resistant direct child/grandchild group killed, direct wrapper waited, unrelated sentinel survives')
            pids.unlink()
            driver=d/'driver.py'
            driver.write_text("import importlib.util,time,sys\ns=importlib.util.spec_from_file_location('h',sys.argv[1]);h=importlib.util.module_from_spec(s);s.loader.exec_module(h)\nh.DEADLINE=time.time()+31\nh.CHILD=h.spawn([sys.executable,sys.argv[2],sys.argv[3]])\nwhile True:time.sleep(.1)\n")
            parent=subprocess.Popen([sys.executable,str(driver),str(STATE/'archive_G20T1.py'),str(family),str(pids)])
            until(pids.exists)
            owned=list(map(int,pids.read_text().split()))
            parent.kill();parent.wait(timeout=2)
            until(lambda: all(not alive(p) for p in owned),24)
            assert sentinel.poll() is None
            checks.append('independent child timeout kills resistant descendants after abrupt driver death; no helper cleanup assumed')
        finally:
            h.cleanup_child(grace=.1)
            for p in locals().get('owned',[]):
                if alive(p):os.kill(p,signal.SIGKILL)
            sentinel.terminate();sentinel.wait(timeout=2)
    preview=subprocess.run([sys.executable,str(STATE/'archive_G20T1.py'),'--plan',str(STATE/'plan.json'),'--pool',plan['pools'][0]['name']],capture_output=True,text=True,check=True,timeout=5)
    assert json.loads(preview.stdout)['mode']=='PLAN_ONLY'
    assert not (STATE/'staging').exists() and not (STATE/'run').exists()
    checks.append('actual helper CLI plan-only accepts and creates no run/staging')
    print(json.dumps({'status':'PASS','seconds':time.time()-started,'checks':checks,'scope':'Disposable children, small pinned files and CLI parsing only; no actual dataset inventory, archive, transfer or process-wide scan.'},indent=2))

if __name__=='__main__':main()
