from __future__ import annotations
import concurrent.futures
import collections
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import time
import chess
import zstandard

OUT = Path(__file__).resolve().parent
PREREG = OUT / 'preregistration.json'
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stat(p):
    s = Path(p).stat()
    return {k:getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}
class HashReader:
    def __init__(self, f): self.f=f; self.h=hashlib.sha256()
    def read(self, n=-1):
        b=self.f.read(n); self.h.update(b); return b

def inspect(row):
    reasons=[]; metrics=collections.Counter(rows=1)
    noresult=row.get('result') is None
    metrics['no_result' if noresult else 'has_result']+=1
    board=chess.Board(row['fen'])
    legal={m.uci() for m in board.legal_moves}
    if not board.is_valid(): reasons.append('invalid_fen_board')
    phases=row.get('phases',[])
    blocks=[b for b in phases[0]['per_depth'] if int(b['depth'])==9] if phases else []
    metrics['phase0_d9_block_count_'+str(len(blocks))]+=1
    if len(blocks)!=1: reasons.append('phase0_d9_block_count')
    block=blocks[-1] if blocks else {}
    lines=block.get('lines',[])
    complete=block.get('complete') is True
    metrics['phase0_d9_complete' if complete else 'phase0_d9_not_complete']+=1
    if not complete: reasons.append('phase0_d9_not_complete')
    ranks=[int(x[0]) for x in lines]; moves=[str(x[1]) for x in lines]
    counts=collections.Counter(moves); missing=sorted(legal-set(moves)); extra=sorted(set(moves)-legal)
    rankok=sorted(ranks)==list(range(1,len(legal)+1))
    if not rankok: reasons.append('rank_ids_not_exact_legal_width')
    if len(set(moves))!=len(moves): reasons.append('duplicate_moves')
    if missing: reasons.append('missing_legal_moves')
    if extra: reasons.append('extra_illegal_moves')
    if not all(math.isfinite(float(x[2])) for x in lines): reasons.append('phase0_d9_nonfinite_score')
    if phases and (phases[0].get('searchmoves') is not None or phases[0].get('width_requested')!='all'):
        reasons.append('phase0_not_full_width_protocol')
    metrics['phase0_d9_rank_ids_exact']+=rankok
    metrics['phase0_d9_duplicate_move_rows']+=len(set(moves))!=len(moves)
    metrics['phase0_d9_missing_support_rows']+=bool(missing)
    metrics['phase0_d9_extra_support_rows']+=bool(extra)
    policy_ok=not reasons
    metrics['phase0_d9_structural_ok']+=policy_ok
    metrics['result_and_phase0_d9_structural_ok']+=policy_ok and not noresult
    base={str(x[1]):float(x[2]) for x in lines}
    latest={m:(cp,0) for m,cp in base.items()}
    later_blocks=0
    for pi,phase in enumerate(phases[1:],1):
        d9=[b for b in phase['per_depth'] if int(b['depth'])==9]
        if d9:
            later_blocks+=1
            for line in d9[-1]['lines']:
                if str(line[1]) in base: latest[str(line[1])]=(float(line[2]),pi)
    metrics['later_phase_d9_present']+=bool(later_blocks)
    metrics['later_phase_observes_any_base_move']+=any(pi>0 for cp,pi in latest.values())
    metrics['later_phase_changes_any_base_cp']+=any(cp!=base[m] for m,(cp,pi) in latest.items())
    value_ok=bool(latest) and all(math.isfinite(cp) for cp,pi in latest.values())
    if not value_ok: reasons.append('latest_phase_value_nonfinite_or_missing')
    if value_ok:
        # Original order tie-break agrees with frozen apply_scheme/np.argmax.
        best=max(moves,key=lambda m:latest[m][0])
        metrics['latest_best_phase_'+str(latest[best][1])]+=1
        metrics['latest_best_cp_differs_from_phase0_best']+=latest[best][0]!=max(base.values())
    metrics['result_and_policy_and_value_structural_ok']+=policy_ok and value_ok and not noresult
    for reason in reasons: metrics['failure_'+reason]+=1
    detail={'reasons':reasons,'has_result':not noresult,'legal_count':len(legal),'complete':complete,'line_count':len(lines),'unique_moves':len(counts),'missing_legal':missing,'extra_illegal':extra,'duplicate_moves':{m:n for m,n in counts.items() if n>1},'later_phase_d9_blocks':later_blocks}
    return metrics, detail

def worker(source):
    start=time.monotonic(); counts=collections.Counter(); shards=[]; examples=[]
    failure_path=OUT/(source['source_id']+'.failures.jsonl')
    noresult_path=OUT/(source['source_id']+'.no_result.jsonl')
    with failure_path.open('x') as failures, noresult_path.open('x') as excluded:
        for selected in source['selection']:
            p=Path(source['source_dir'])/selected['source_shard']; before=stat(p); c=collections.Counter(); n=0
            assert before['st_size']==selected['raw_bytes']
            with p.open('rb') as f:
                hashed=HashReader(f)
                with zstandard.ZstdDecompressor().stream_reader(hashed) as zr:
                    for physical,line in enumerate(io.TextIOWrapper(zr,encoding='utf-8')):
                        row=json.loads(line); n+=1
                        m,d=inspect(row); c.update(m)
                        ref={'source_id':source['source_id'],'source_dir':source['source_dir'],'source_shard':p.name,'source_row':physical,'worker_id':row.get('worker_id'),'game_id':row.get('game_id'),'ply':row.get('ply'),'input_key':row.get('input_key')}
                        if row.get('result') is None: excluded.write(json.dumps(ref,separators=(',',':'))+'\n')
                        if d['reasons']:
                            failures.write(json.dumps({**ref,**d},separators=(',',':'))+'\n')
                            if len(examples)<5: examples.append({**ref,**d})
                digest=hashed.h.hexdigest()
            assert n==selected['raw_rows'], (p,n,selected['raw_rows'])
            assert digest==selected['source_sha256'], (p,digest)
            assert stat(p)==before, ('source changed',p)
            counts.update(c)
            shard={'path':str(p),'sha256':digest,'stable_stat':before,'counts':dict(c)}; shards.append(shard)
            with (OUT/(source['source_id']+'.progress.jsonl')).open('a') as progress: progress.write(json.dumps(shard)+'\n')
            print(source['source_id'],p.name,n,'cumulative',counts['rows'],'support_failures',counts['phase0_d9_missing_support_rows'],flush=True)
    ru=resource.getrusage(resource.RUSAGE_SELF)
    return {'source_id':source['source_id'],'counts':dict(counts),'shards':shards,'examples':examples,'wall_seconds':time.monotonic()-start,'user_seconds':ru.ru_utime,'system_seconds':ru.ru_stime,'max_rss_kib':ru.ru_maxrss,'failures_sha256':sha(failure_path),'no_result_sha256':sha(noresult_path)}

def main():
    started=time.time(); plan=json.loads(PREREG.read_text())
    assert sha(__file__)==plan['script_sha256']
    for pin in plan['pins']: assert sha(pin['path'])==pin['sha256'],pin
    registration=json.loads(Path(plan['original_registration']['path']).read_text())
    assert sum(s['raw_rows'] for s in registration['sources'])==531412
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool: results=list(pool.map(worker,registration['sources']))
    for pin in plan['pins']: assert sha(pin['path'])==pin['sha256'],pin
    totals=collections.Counter()
    for r in results: totals.update(r['counts'])
    readout={'schema':1,'status':'COMPLETE_STRUCTURAL_CENSUS_ONLY','started_unix':started,'ended_unix':time.time(),'wall_seconds':time.time()-started,'preregistration_sha256':sha(PREREG),'script_sha256':sha(__file__),'counts':dict(totals),'sources':results,'limitations':plan['limitations'],'runtime':{'python':os.sys.version,'chess':chess.__version__,'zstandard':zstandard.__version__,'affinity':sorted(os.sched_getaffinity(0)),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')}}
    (OUT/'readout.json').open('x').write(json.dumps(readout,indent=2)+'\n')
    print('COMPLETE',json.dumps(dict(totals)),flush=True)
if __name__=='__main__': main()
