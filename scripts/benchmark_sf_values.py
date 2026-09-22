"""Matched-row SF scalar-value cost screen; never changes training targets."""
from __future__ import annotations
import argparse,hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import benchmark_sf_generation as bounds
require=bounds.require

def validate(plan):
    require(plan['profile']=='matched-root-value-cost-v1','profile differs')
    require(plan['arms']==[['root_d8',8,1],['root_d10',10,1],['all_d8',8,'all']],'arm contract differs')
    require(plan['rows'] in (64,128),'sample size differs')
    require(0<plan['cpu_seconds']<=1500 and 0<plan['wall_seconds']<=1800,'budget too large')
    require(0<len(set(plan['affinity']))<=2 and plan['memory_gib']>=40 and plan['disk_gib']>=100,'resource limits differ')
    require(plan['hash_mb']==64 and plan['threads']==1 and plan['output_bytes']<=128*2**20,'engine/output limits differ')
    require(subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==plan['runtime_head'],'runtime changed')
    subprocess.run(['git','-C',str(ROOT),'diff','--exit-code','HEAD','--'],check=True)
    for pin in plan['pins']:require(bounds.sha(Path(pin['path']))==pin['sha256'],'pin changed '+pin['path'])

def worker(plan):
    from scripts import gen_sf_rooted_corpus as corpus,derive_corpus_targets as derive
    from chess_anti_engine.stockfish.uci import StockfishUCI
    samples=[json.loads(s) for s in Path(plan['sample']).read_text().splitlines()]
    require(len(samples)==plan['rows'],'sample row count differs')
    engines={};searchers={};out=Path(plan['out']);timings={a[0]:[] for a in plan['arms']}
    try:
        for name,depth,width in plan['arms']:
            engine=StockfishUCI(plan['stockfish'],multipv=1,hash_mb=64,threads=1,nice=19,syzygy_path=plan['syzygy_path'],read_timeout_s=30)
            engines[name]=engine
            searchers[name]=corpus.StaircaseSearcher(engine=engine,staircase=corpus.parse_staircase('1:8'),cp_slope=.006,cp_draw_width=120,search_timeout_s=15)
        with (out/'labels.jsonl').open('x') as stream:
            for index,sample in enumerate(samples):
                row=sample['row'];raw=json.dumps(row,sort_keys=True,separators=(',',':')).encode();require(hashlib.sha256(raw).hexdigest()==sample['row_sha256'],'sample identity changed')
                board=derive.board_from_row(row);require(board.is_valid() and board.legal_moves.count()>0,'invalid sample board')
                history=corpus.RowHistory(fen=row['fen'],root_fen=row['history_root_fen'],uci=tuple(row['history_uci']),reason=row['history_root_reason'])
                # Rotate arm order per row. Each independent engine clears its
                # own TT for every position; no cross-arm/row warm-table reuse.
                order=plan['arms'][index%3:]+plan['arms'][:index%3]
                for name,depth,width in order:
                    engine=engines[name];searcher=searchers[name];commands=[];send=engine._send
                    def capture(command):commands.append(command);return send(command)
                    engine._send=capture;began=time.monotonic()
                    try:
                        searcher.new_game();realized=board.legal_moves.count() if width=='all' else 1
                        lines=searcher.stream(history,depth=depth,multipv=realized)
                    finally:engine._send=send
                    elapsed=time.monotonic()-began;parsed=corpus.parse_depth_blocks(lines,expected_lines=realized);block,full=corpus.deepest_block_with_width(parsed.blocks,want=realized)
                    require(full and block.complete and block.depth==depth,'incomplete requested search')
                    require(len({p.move for p in block.lines})==realized,'duplicate search moves')
                    require(all(p.move in {m.uci() for m in board.legal_moves} for p in block.lines),'illegal search result')
                    require(corpus.position_command(history) in commands and f'go depth {depth}' in commands,'effective search command missing')
                    record={'sample_index':index,'source':sample['source'],'source_row_index':sample['source_row_index'],'row_sha256':sample['row_sha256'],'game_id':row['game_id'],'ply':row['ply'],'arm':name,'elapsed_seconds':elapsed,'depth':depth,'multipv':realized,'nodes':block.nodes_at_depth,'commands':commands,'position_command':corpus.position_command(history),'pv':[{'rank':p.rank,'move':p.move,'effective_cp':p.effective_cp,'nodes':p.nodes} for p in block.lines],'raw_info':lines}
                    stream.write(json.dumps(record,separators=(',',':'))+'\n');stream.flush();timings[name].append(elapsed)
        summary={'status':'COMPLETE_MATCHED_VALUE_COST','rows':len(samples),'arms':{name:{'labels':len(values),'search_and_reset_seconds':sum(values),'labels_per_search_second':len(values)/sum(values),'max_seconds':max(values),'median_seconds':sorted(values)[len(values)//2]} for name,values in timings.items()},'interpretation':'Same existing history-bearing positions; scalar versus full-width label cost only. No generation rate, value-quality or Elo claim.'}
        bounds.dump(out/'worker.complete.json',summary)
    finally:
        for engine in engines.values():engine.close()

def execute(plan):
    import shutil
    out=Path(plan['out']);require(not out.exists(),'fresh output required')
    require(not any(Path(p).exists() for p in plan['stop_paths']),'STOP')
    require(bounds.memory()>=plan['memory_gib']*2**30,'RAM reserve');require(shutil.disk_usage(out.parent).free>=plan['disk_gib']*2**30,'disk reserve')
    os.sched_setaffinity(0,plan['affinity']);os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    out.mkdir();started=time.monotonic();base=resource.getrusage(resource.RUSAGE_SELF);baseline=bounds.child_baseline();child=None;owned=None
    def interrupt(sig,frame):raise InterruptedError(f'signal {sig}')
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,interrupt)
    try:
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        argv=[sys.executable,str(Path(__file__).resolve()),'--plan',str(plan['_path']),'--sha256',plan['_sha'],'--worker']
        with (out/'worker.log').open('x') as log:
            child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);owned=bounds.OwnedProcesses(child.pid,baseline)
            while child.poll() is None:
                cpu=owned.sample();r=resource.getrusage(resource.RUSAGE_SELF);controller=r.ru_utime+r.ru_stime-base.ru_utime-base.ru_stime
                require(cpu+controller<plan['cpu_seconds'],'CPU budget');require(time.monotonic()-started<plan['wall_seconds'],'wall budget')
                require(bounds.memory()>=plan['memory_gib']*2**30,'RAM reserve');require(shutil.disk_usage(out).free>=96*2**30,'disk reserve')
                require(bounds.output_bytes(out)<plan['output_bytes']*.75,'output stop margin');require(not any(Path(p).exists() for p in plan['stop_paths']),'STOP')
                time.sleep(1)
            require(child.returncode==0,'worker failed')
        require((out/'worker.complete.json').exists(),'completion missing')
        owned.stop(child)
        bounds.dump(out/'complete.json',{'status':'COMPLETE_BOUNDED_VALUE_COST','wall_seconds':time.monotonic()-started,'observed_child_cpu_seconds':owned.sample(),'worker':json.loads((out/'worker.complete.json').read_text())})
    except BaseException as exc:
        bounds.dump(out/'failed.json',{'status':'FAILED_VALUE_COST','error':repr(exc)});raise
    finally:
        if child is not None and owned is not None:owned.stop(child)

def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--sha256',required=True);parser.add_argument('--execute',action='store_true');parser.add_argument('--worker',action='store_true');args=parser.parse_args();require(bounds.sha(args.plan)==args.sha256,'plan changed');plan=json.loads(args.plan.read_text());validate(plan)
    if args.worker:worker(plan)
    elif args.execute:execute({**plan,'_path':str(args.plan),'_sha':args.sha256})
    else:print(json.dumps({'status':'VALID_NOT_LAUNCHED','rows':plan['rows']}))
if __name__=='__main__':main()
