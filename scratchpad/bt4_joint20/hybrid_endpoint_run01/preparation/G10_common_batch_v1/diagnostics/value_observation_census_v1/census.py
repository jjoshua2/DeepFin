from __future__ import annotations
import collections
import concurrent.futures
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import resource
import time
import chess
import zstandard
OUT=Path(__file__).resolve().parent
PREVIOUS=OUT.parent/'structural_census_v1'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stat(p):
    s=Path(p).stat()
    return {k:getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}
class HashReader:
    def __init__(self,f): self.f=f;self.h=hashlib.sha256()
    def read(self,n=-1):
        b=self.f.read(n);self.h.update(b);return b

def inspect(row,policy_valid):
    counts=collections.Counter(rows=1)
    subset=policy_valid and row.get('result') is not None
    counts['result_and_phase0_policy_valid_rows']+=subset
    legal={m.uci() for m in chess.Board(row['fen']).legal_moves}
    baseblock=[b for b in row['phases'][0]['per_depth'] if int(b['depth'])==9][-1]
    base={str(x[1]):float(x[2]) for x in baseblock['lines']}
    final={m:(cp,0) for m,cp in base.items()}
    phase_data=[];flags=set();details=[]
    for pi,phase in enumerate(row['phases'][1:],1):
        blocks=[b for b in phase['per_depth'] if int(b['depth'])==9]
        if not blocks:continue
        block=blocks[-1] # RowBank overwrites any earlier same-depth block.
        lines=block['lines'];width=int(phase['width_realized']);ranks=[x[0] for x in lines]
        rank_bad=any(type(r) is not int or r<1 or r>width for r in ranks) or len(set(ranks))!=len(ranks)
        exact_ranks=sorted(ranks)==list(range(1,width+1))
        claimed_bad=bool(block['complete']) and not exact_ranks
        grouped=collections.defaultdict(list)
        for line in lines:grouped[str(line[1])].append(float(line[2]))
        duplicates={m:vs for m,vs in grouped.items() if len(vs)>1}
        conflicting={m for m,vs in duplicates.items() if any(v!=vs[0] for v in vs[1:])}
        agreeing=set(duplicates)-conflicting
        illegal=set(grouped)-legal
        searchmoves=phase.get('searchmoves')
        outside=set(grouped)-set(searchmoves) if searchmoves is not None else set()
        nonfinite={m for m,vs in grouped.items() if any(not math.isfinite(v) for v in vs)}
        phase_data.append((pi,grouped,conflicting,agreeing,illegal,outside,nonfinite,rank_bad or claimed_bad))
        for m,vs in grouped.items():
            if m in base:final[m]=(vs[-1],pi)
        counts['later_d9_blocks']+=1
        counts['later_d9_complete_false_blocks']+=not block['complete']
        counts['later_d9_not_full_rank_roster_blocks']+=not exact_ranks
        counts['later_d9_missing_full_legal_roster_blocks']+=bool(legal-set(grouped))
        checks={'duplicate_agreeing':agreeing,'duplicate_conflicting':conflicting,'illegal_extra':illegal,'outside_searchmoves':outside,'nonfinite':nonfinite,'malformed_rank_ids':rank_bad,'claimed_complete_rank_mismatch':claimed_bad,'duplicate_depth_blocks':len(blocks)>1}
        for name,v in checks.items():
            if v: flags.add('any_later_'+name);counts['blocks_'+name]+=1
        if any(checks.values()):
            details.append({'phase':pi,'width':width,'complete':block['complete'],'rank_ids':ranks,'duplicates':duplicates,'illegal_extra':sorted(illegal),'outside_searchmoves':sorted(outside),'nonfinite_moves':sorted(nonfinite),'malformed_rank_ids':rank_bad,'claimed_complete_rank_mismatch':claimed_bad,'duplicate_depth_blocks':len(blocks)>1})
    for pi,grouped,conflicting,agreeing,illegal,outside,nonfinite,badranks in phase_data:
        used={m for m,(cp,p) in final.items() if p==pi}
        usedchecks={'duplicate_conflicting':used&conflicting,'duplicate_agreeing':used&agreeing,'illegal_extra':used&illegal,'outside_searchmoves':used&outside,'nonfinite_emission':used&nonfinite,'malformed_rank_block':bool(used) and badranks}
        for name,v in usedchecks.items():
            if v:flags.add('consumed_'+name)
        if conflicting-used:flags.add('shadowed_or_outside_base_conflicting_duplicate')
        for detail in details:
            if detail['phase']==pi:
                detail['consumed_moves']=sorted(used)
                detail['consumed_conflicting_duplicate_moves']=sorted(used&conflicting)
    if any(not math.isfinite(cp) for cp,pi in final.values()):flags.add('final_nonfinite_value')
    if final and all(math.isfinite(cp) for cp,pi in final.values()):
        best=max(base,key=lambda m:final[m][0]);winner_phase=final[best][1]
        for pi,grouped,conflicting,agreeing,illegal,outside,nonfinite,badranks in phase_data:
            if pi==winner_phase and best in conflicting:flags.add('selected_best_conflicting_duplicate')
    problematic={'consumed_duplicate_conflicting','consumed_illegal_extra','consumed_outside_searchmoves','consumed_nonfinite_emission','consumed_malformed_rank_block','final_nonfinite_value'}
    if flags&problematic:flags.add('consumed_value_issue_union')
    for flag in flags:
        counts['rows_'+flag]+=1
        if subset:counts['qualified_rows_'+flag]+=1
    return counts,{'has_result':row.get('result') is not None,'phase0_policy_valid':policy_valid,'result_and_phase0_policy_valid':subset,'flags':sorted(flags),'anomalous_phases':details}

def worker(source):
    start=time.monotonic();counts=collections.Counter();shards=[];examples=[]
    policy_failures={x['source_row'] for x in []}
    badrefs=set()
    for line in (PREVIOUS/(source['source_id']+'.failures.jsonl')).read_text().splitlines():
        r=json.loads(line);badrefs.add((r['source_shard'],r['source_row']))
    anomalies=OUT/(source['source_id']+'.anomalies.jsonl')
    with anomalies.open('x') as records:
        for selected in source['selection']:
            p=Path(source['source_dir'])/selected['source_shard'];before=stat(p);n=0;c=collections.Counter()
            assert before['st_size']==selected['raw_bytes']
            with p.open('rb') as f:
                hr=HashReader(f)
                with zstandard.ZstdDecompressor().stream_reader(hr) as zr:
                    for physical,line in enumerate(io.TextIOWrapper(zr,encoding='utf-8')):
                        row=json.loads(line);n+=1;m,d=inspect(row,(p.name,physical) not in badrefs);c.update(m)
                        if d['flags']:
                            ref={'source_id':source['source_id'],'source_dir':source['source_dir'],'source_shard':p.name,'source_row':physical,'worker_id':row.get('worker_id'),'game_id':row.get('game_id'),'ply':row.get('ply'),'input_key':row.get('input_key')}
                            detail={**ref,**d};records.write(json.dumps(detail,separators=(',',':'))+'\n')
                            if len(examples)<8:examples.append(detail)
                digest=hr.h.hexdigest()
            assert digest==selected['source_sha256'],p
            assert n==selected['raw_rows'] and stat(p)==before,p
            counts.update(c);shards.append({'path':str(p),'sha256':digest,'stable_stat':before,'counts':dict(c)})
            print(source['source_id'],p.name,'rows',n,flush=True)
    ru=resource.getrusage(resource.RUSAGE_SELF)
    return {'source_id':source['source_id'],'counts':dict(counts),'shards':shards,'examples':examples,'anomalies_sha256':sha(anomalies),'wall_seconds':time.monotonic()-start,'user_seconds':ru.ru_utime,'system_seconds':ru.ru_stime,'max_rss_kib':ru.ru_maxrss}

def main():
    start=time.time();prereg=OUT/'preregistration.json';plan=json.loads(prereg.read_text())
    assert sha(__file__)==plan['script_sha256']
    for pin in plan['pins']:assert sha(pin['path'])==pin['sha256'],pin
    registration=json.loads(Path(plan['original_registration']['path']).read_text())
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(worker,registration['sources']))
    for pin in plan['pins']:assert sha(pin['path'])==pin['sha256'],pin
    total=collections.Counter()
    for result in results:total.update(result['counts'])
    assert total['rows']==531412 and total['result_and_phase0_policy_valid_rows']==528545,total
    report={'schema':1,'status':'COMPLETE_BOUNDED_VALUE_OBSERVATION_CENSUS','preregistration_sha256':sha(prereg),'script_sha256':sha(__file__),'counts':dict(total),'sources':results,'started_unix':start,'ended_unix':time.time(),'wall_seconds':time.time()-start,'runtime':{'python':os.sys.version,'chess':chess.__version__,'zstandard':zstandard.__version__,'affinity':sorted(os.sched_getaffinity(0)),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')},'limitations':plan['limitations']}
    (OUT/'readout.json').open('x').write(json.dumps(report,indent=2)+'\n');print('COMPLETE',dict(total),flush=True)
if __name__=='__main__':main()
