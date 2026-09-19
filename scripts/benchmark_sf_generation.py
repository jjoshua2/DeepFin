"""Bounded CPU corpus-throughput screen; fresh namespaces, closed games only.

A plan supplies exact generation commands, source pins and limits. Default mode
validates without launching. No active corpus is resumed or mutated.
"""
from __future__ import annotations
import argparse,hashlib,json,os,resource,shutil,signal,subprocess,time
from pathlib import Path
from typing import Any

def require(ok: bool,message: str)->None:
    if not ok:raise ValueError(message)
def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1<<20),b''):h.update(part)
    return h.hexdigest()
def dump(path:Path,value:Any)->None:
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def memory()->int:
    return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
def snapshot()->dict[int,dict[str,Any]]:
    result={}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            fields=(path/'stat').read_text().rsplit(')',1)[1].split()
            result[int(path.name)]={'parent':int(fields[1]),'start':int(fields[19]),'ticks':int(fields[11])+int(fields[12]),'state':fields[0]}
        except (FileNotFoundError,ProcessLookupError,PermissionError,IndexError):continue
    return result
class OwnedProcesses:
    """Track escaped engine sessions by ancestry and guard against PID reuse."""
    def __init__(self,pid:int):
        self.root=pid;self.root_start=snapshot().get(pid,{}).get('start');self.known:dict[int,int]={};self.peaks:dict[tuple[int,int],int]={}
    def sample(self)->float:
        current=snapshot();selected={self.root} if self.root in current and current[self.root]['start']==self.root_start else set()
        selected.update(pid for pid,start in self.known.items() if pid in current and current[pid]['start']==start)
        while True:
            extra={pid for pid,item in current.items() if item['parent'] in selected}
            if extra<=selected:break
            selected|=extra
        for pid in selected:
            item=current[pid];self.known[pid]=item['start'];key=(pid,item['start']);self.peaks[key]=max(self.peaks.get(key,0),item['ticks'])
        return sum(self.peaks.values())/os.sysconf('SC_CLK_TCK')
    def signal(self,sig:int)->None:
        current=snapshot()
        for pid,start in self.known.items():
            if pid in current and current[pid]['start']==start:
                try:os.kill(pid,sig)
                except ProcessLookupError:pass
    def stop(self,child:subprocess.Popen)->None:
        self.sample();self.signal(signal.SIGTERM)
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:pass
        self.sample();self.signal(signal.SIGKILL)
        child.wait(timeout=5)
        time.sleep(.1)
        current=snapshot()
        require(not any(pid in current and current[pid]['start']==start and current[pid]['state']!='Z' for pid,start in self.known.items()),'owned process survived cleanup')
def output_bytes(path:Path)->int:return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
def flag(command:list[str],name:str)->str:
    require(command.count(name)==1,'missing/duplicate '+name);return command[command.index(name)+1]
def validate(plan:dict[str,Any])->None:
    require(plan['status']=='READY_BOUNDED_CPU_SCREEN','plan not ready')
    require(plan['cpu_budget_seconds']<=1500 and plan['wall_budget_seconds']<=1800,'budget too large')
    require(len(plan['affinity'])<=8 and len(set(plan['affinity']))==len(plan['affinity']),'CPU affinity exceeds8')
    require(plan['launch_disk_gib']>=100 and plan['memory_gib']>=40 and plan['output_limit_bytes']<=2*2**30,'resource contract differs')
    require(len(plan['cells'])==6,'expected six comparisons')
    require({(c['policy'],c['concurrency']) for c in plan['cells']}=={(p,n) for p in ['d8','g10'] for n in [1,2,4]},'grid differs')
    for cell in plan['cells']:
        cmd=cell['command'];require('--resume' not in cmd,'fresh corpora only')
        require(flag(cmd,'--out-dir')==str(Path(plan['out'])/cell['id']),'foreign output namespace')
        require(flag(cmd,'--workers')=='4' and flag(cmd,'--worker-concurrency')==str(cell['concurrency']),'worker partition differs')
        require(flag(cmd,'--nice')=='19' and flag(cmd,'--shard-rows')=='256','benchmark execution shape differs')
        require(flag(cmd,'--staircase')==('all:8' if cell['policy']=='d8' else 'all:9,8:10,4:12'),'staircase differs')
        require(flag(cmd,'--staircase-policy')==('fixed' if cell['policy']=='d8' else 'g10'),'gate differs')
        require(0<cell['seconds']<=100,'cell exceeds budget')
    require(subprocess.check_output(['git','-C',plan['runtime'],'rev-parse','HEAD'],text=True).strip()==plan['runtime_head'],'runtime HEAD changed')
    subprocess.run(['git','-C',plan['runtime'],'diff','--exit-code','HEAD','--'],check=True)
    for pin in plan['pins']:require(sha(Path(pin['path']))==pin['sha256'],'changed pin '+pin['path'])

def closed_readout(root:Path,depth:int)->dict[str,Any]:
    # Decoder is production code; no GPU and no derived arrays are written.
    from dataclasses import replace
    from scripts import gen_sf_rooted_corpus as corpus,derive_corpus_targets as derive
    manifest=json.loads((root/'manifest.json').read_text())
    records=[]
    for path in sorted(root.glob('w*.progress.jsonl')):
        lines=path.read_text().splitlines()
        for index,line in enumerate(lines):
            try:record=json.loads(line)
            except json.JSONDecodeError:
                require(index==len(lines)-1,'nonfinal torn progress record');continue
            records.append(record)
    # Progress schema is inspected strictly; only referenced closed shards count.
    shards=[]
    for record in records:
        if record.get('kind')=='shard':shards.append(record)
    if records and not shards:
        # Current writer records shard fields directly, without a kind discriminator.
        shards=[r for r in records if r.get('path') is not None and r.get('rows',0)>0]
    expected=[{'width':'all','depth':8}] if depth==8 else [{'width':'all','depth':9},{'width':8,'depth':10},{'width':4,'depth':12}]
    require(manifest['staircase_parsed']==expected,'realized staircase differs')
    require(manifest['staircase_gate']['policy']==('fixed' if depth==8 else 'g10'),'realized gate differs')
    require(len({e['path'] for e in shards})==len(shards),'duplicate closed shard')
    scheme=replace(derive.parse_scheme(f'uniform-d{depth}'),policy_observation='phase0',value_observation='latest-phase')
    inspector=derive.TargetDeriver(derive.DeriveOptions(scheme,.0005,1.,1.,0,0,8192,0))
    counts={'banked_rows':0,'eligible_rows':0,'no_result_rows':0,'invalid_rows':0,'closed_shards':len(shards)};failures={}
    for entry in shards:
        path=root/entry['path'];require(path.is_file() and path.resolve().parent==root.resolve(),'closed shard absent/foreign');seen=0
        for row in corpus.iter_shard_rows(path):
            seen+=1;counts['banked_rows']+=1
            derive._check_row_identity(row,manifest['config_sha256'])
            if row.get('result') is None:counts['no_result_rows']+=1;continue
            try:
                board=inspector._board_for(row);require(board.is_valid(),'invalid board')
                inspector._verify_input_key(row,inspector._encode(board),count_emitted=False)
                inspector._validate_selected_policy_block(row,board)
                values=derive.apply_scheme(derive.RowBank(row),scheme);inspector._check_support(board,values,row)
                selected=inspector._value_view(derive.RowBank(row),values);require(set(selected.moves)=={m.uci() for m in board.legal_moves},'value support differs')
                derive.wdl_target_from_result(row['result'])
            except (ValueError,KeyError,IndexError,TypeError,derive.CorpusIntegrityError,derive.EnvelopeMiss) as exc:
                counts['invalid_rows']+=1;kind=type(exc).__name__+':'+str(exc)[:120];failures[kind]=failures.get(kind,0)+1
            else:counts['eligible_rows']+=1
        require(seen==entry['rows'],'closed shard rowcount differs')
    return {**counts,'failures':failures,'manifest_sha256':sha(root/'manifest.json'),'scope':'Closed-game rows passing result, stored-input, phase0 policy and latest-phase value support checks; no cross-corpus dedup or teacher labeling.'}

def execute(plan:dict[str,Any])->dict[str,Any]:
    validate(plan);os.sched_setaffinity(0,plan['affinity']);os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    for key,value in plan['env'].items():os.environ[key]=value
    os.environ['CUDA_VISIBLE_DEVICES']=''
    root=Path(plan['out']);require(not root.exists(),'fresh benchmark output required')
    require(shutil.disk_usage(root.parent).free>=plan['launch_disk_gib']*2**30,'launch disk reserve')
    require(memory()>=plan['memory_gib']*2**30,'memory reserve')
    root.mkdir();start=time.monotonic();cpu_used=0.;results=[];owned=None;child=None
    usage=resource.getrusage(resource.RUSAGE_SELF);controller_start=usage.ru_utime+usage.ru_stime
    def controller_cpu():
        usage=resource.getrusage(resource.RUSAGE_SELF);return usage.ru_utime+usage.ru_stime-controller_start
    def stop(sig,frame):raise InterruptedError(f'signal{sig}')
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
    try:
        for cell in plan['cells']:
            require(shutil.disk_usage(root).free>=plan['launch_disk_gib']*2**30,'next-cell disk reserve')
            require(cpu_used+controller_cpu()<plan['cpu_budget_seconds'],'CPU budget exhausted')
            began=time.monotonic();out=root/cell['id'];samples=[]
            env={**os.environ,**plan['env']};env['CUDA_VISIBLE_DEVICES']=''
            def setup():os.sched_setaffinity(0,plan['affinity']);os.nice(19)
            with (root/(cell['id']+'.log')).open('x') as stream:
                child=subprocess.Popen(cell['command'],cwd=plan['runtime'],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=setup);owned=OwnedProcesses(child.pid)
                reason='natural_completion'
                while child.poll() is None:
                    used=owned.sample();now=time.monotonic();size=output_bytes(root)
                    require(memory()>=plan['memory_gib']*2**30,'memory reserve')
                    require(shutil.disk_usage(root).free>=96*2**30,'running disk reserve')
                    require(not (root/'STOP').exists() and not (root.parent/'STOP').exists(),'STOP')
                    require(now-start<plan['wall_budget_seconds'],'wall budget')
                    require(size<plan['output_limit_bytes']*.75,'output stop margin')
                    samples.append({'elapsed':now-began,'cpu_seconds':used,'bytes':size})
                    if cpu_used+used+controller_cpu()>=plan['cpu_budget_seconds']:reason='cpu_budget';break
                    if now-began>=cell['seconds']:reason='cell_budget';break
                    time.sleep(1)
                owned.stop(child);used=owned.sample();cpu_used+=used;elapsed=time.monotonic()-began
            require(output_bytes(root)<=plan['output_limit_bytes'],'output cap exceeded')
            require(reason!='natural_completion' or child.returncode==0,'generator failed before bounded stop')
            bank=closed_readout(out,8 if cell['policy']=='d8' else 9)
            result={'id':cell['id'],'policy':cell['policy'],'concurrency':cell['concurrency'],'elapsed_seconds':elapsed,'cpu_seconds_observed':used,'stop_reason':reason,'returncode':child.returncode,'samples':samples,**bank,'eligible_rows_per_wall_second':bank['eligible_rows']/elapsed,'banked_rows_per_wall_second':bank['banked_rows']/elapsed}
            dump(root/(cell['id']+'.result.json'),result);results.append(result);owned=None;child=None
            if reason=='cpu_budget' or cpu_used+controller_cpu()>=plan['cpu_budget_seconds']:break
        report={'status':'COMPLETE_SCREEN' if len(results)==6 else 'BOUNDED_PARTIAL_SCREEN','cells':results,'cpu_seconds_observed':cpu_used,'controller_cpu_seconds':controller_cpu(),'elapsed_seconds':time.monotonic()-start,'limitations':['Short cold-start closed-shard rates are conservative and may have high opening/game-length variance.','Proc CPU accounting may miss very short children;1500s observed cap reserves300s under30core-minute allocation.','No strength conclusion, no production corpus mutation, and no500M rate guarantee.']};dump(root/'complete.json',report);return report
    except BaseException as exc:
        dump(root/'failed.json',{'status':'FAILED_BOUNDED_SCREEN','error':repr(exc),'completed_cells':results,'cpu_seconds_observed':cpu_used,'elapsed_seconds':time.monotonic()-start})
        raise
    finally:
        if owned is not None and child is not None:owned.stop(child)

def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--expected-sha256',required=True);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    require(sha(args.plan)==args.expected_sha256,'plan pin differs');plan=json.loads(args.plan.read_text());validate(plan)
    if args.execute:print(json.dumps(execute(plan),indent=2))
    else:print(json.dumps({'status':'VALID_NOT_LAUNCHED','cells':len(plan['cells'])}))
if __name__=='__main__':main()
