"""Once-only geometry of the existing 128-row training bank; no inference."""
from __future__ import annotations
import ast
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, cast
import numpy as np

OUT = Path(__file__).resolve().parent
BASE = Path('/home/josh/projects/chess/scratchpad/tailrl_bootstrap_v1')
MAIN = Path('/tmp/deepfin-bt4-readiness-handoff')
sys.path.insert(0, str(MAIN))
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE, FULL_TO_COMPACT_POLICY, uci_to_policy_index
from chess_anti_engine.stockfish.wdl import cp_to_wdl_array, SF_CP_CLAMP_CP


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_functions(path, names):
    """Execute only named original definitions, retaining their exact AST."""
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), globals())


INVALID_INDEX = np.iinfo(np.uint16).max
GLOBAL_SCOPES = {'global', 'c20-global'}
exact_functions(MAIN/'scripts/sf_d9_rank_sidecar.py', ['RankObservation', 'd9_lines', 'rank_observation'])
exact_functions(MAIN/'scripts/bt4_policy_mix.py', ['_source_candidate_set', 'validate_near_max_ratio', 'validate_sf_rank_cap', 'validate_sf_cp_window'])
exact_functions(MAIN/'scripts/derive_corpus_targets.py', ['softmax_at_temp', 'validate_temp', 'shard_stored'])


def normalize(p):
    p = np.asarray(p, dtype=np.float64)
    assert np.isfinite(p).all() and np.all(p >= 0) and p.sum() > 0
    return p / p.sum()


def metrics(p, selected, best):
    p = normalize(p)
    positive = p > 0
    return {'entropy_nats': float(-np.sum(p[positive]*np.log(p[positive]))),
            'top1_mass': float(p.max()), 'top1_ge_099': bool(p.max() >= .99),
            'tail_mass': float(1-p.max()), 'outside_C_mass': float(p[~selected].sum()),
            'raw_cp_maxima_mass': float(p[best].sum()), 'support': int(positive.sum())}


def summarize(rows):
    labels = list(rows[0]['targets']) if rows else []
    return {'rows':len(rows), 'selected_size_histogram':dict(Counter(str(r['selected_size']) for r in rows)),
            'stored_max_count_histogram':dict(Counter(str(r['stored_max_count']) for r in rows)),
            'q_flat_count':sum(r['q_flat'] for r in rows),
            'q_exact_flat_count':sum(r['q_exact_flat'] for r in rows),
            'stored_maxima_have_cp_spread_count':sum(r['stored_maxima_cp_spread'] > 0 for r in rows),
            'selected_larger_than_stored_maxima_count':sum(r['selected_size'] > r['stored_max_count'] for r in rows),
            'targets':{label:{metric:float(np.mean([r['targets'][label][metric] for r in rows]))
                              for metric in rows[0]['targets'][label]} for label in labels}}


def main():
    started = time.time()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.sched_getaffinity(0) == {0,1} and os.getpriority(os.PRIO_PROCESS,0) == 19
    assert not (OUT/'readout.json').exists() and not (OUT/'failed.json').exists()
    plan = json.loads((OUT/'plan.json').read_text())
    pins = dict(plan['pins'])
    pins.update({str(p):sha(p) for p in [Path(__file__),OUT/'plan.json',OUT/'preregistration.md']})
    assert all(sha(p) == h for p,h in pins.items())
    with np.load(BASE/'bank.npz', allow_pickle=False) as f:
        bank = {k:f[k] for k in ['selected_rows','legal_mask','policy_target','effective_cp_d9','reward_d9','game_id','ply_index','source_row_ids','group_ids']}
    with np.load(BASE/'teacher_targets_v1/targets.npz', allow_pickle=False) as f:
        names = f['target_names'].tolist()
        target = f['stored_or_proposed_targets']
        assert np.array_equal(target[names.index('SF')],bank['policy_target'])
        assert np.array_equal(f['selected_rows'],bank['selected_rows'])
        assert np.array_equal(f['legal_mask'],bank['legal_mask'])
        assert np.array_equal(f['source_row_ids'],bank['source_row_ids'])
    raw = [json.loads(line) for line in (BASE/'selected_raw_rows.jsonl').read_text().splitlines()]
    assert len(raw) == len(bank['selected_rows']) == 128
    assert len(set(bank['group_ids'].tolist())) == 43
    records = []
    for i,item in enumerate(raw):
        row = item['raw']; legal = bank['legal_mask'][i].astype(bool)
        assert item['derived_row'] == int(bank['selected_rows'][i])
        assert (int(row['game_id']),int(row['ply'])) == (int(bank['game_id'][i]),int(bank['ply_index'][i]))
        obs = rank_observation(row, top_k=len(d9_lines(row)))
        assert set(obs.indices.tolist()) == set(np.flatnonzero(legal).tolist())
        cp = bank['effective_cp_d9'][i,legal]
        assert np.isfinite(cp).all()
        for line in d9_lines(row):
            idx = int(FULL_TO_COMPACT_POLICY[uci_to_policy_index(str(line[1]),row['stm']=='w')])
            assert bank['effective_cp_d9'][i,idx] == float(line[2])
        sf = bank['policy_target'][i]
        ranks = np.full((1,3), INVALID_INDEX, dtype=np.uint16)
        gaps = np.full((1,3), np.inf, dtype=np.float32)
        k=min(3,obs.count);ranks[0,:k]=obs.indices[:k];gaps[0,:k]=obs.gaps_cp[:k]
        selected = _source_candidate_set(sf[None,:],legal[None,:],scope='sf-cp-window',near_max_ratio=.9,
                    sf_rank_indices=ranks,sf_rank_gaps_cp=gaps,sf_rank_cap=3,sf_cp_window=20)[0,legal]
        storedmax = sf[legal] == sf[legal].max()
        wdl = cp_to_wdl_array(cp,slope=.006,draw_width_cp=120)
        q = wdl[:,0].astype(np.float64)-wdl[:,2].astype(np.float64)
        assert np.max(np.abs((q+1)/2-bank['reward_d9'][i,legal])) < 1e-12
        reconstructed = softmax_at_temp(q,temp=.0005)
        assert np.max(np.abs(reconstructed-sf[legal].astype(np.float64))) <= .0005
        anymate=bool(np.any(np.abs(cp)>SF_CP_CLAMP_CP));best=cp == cp.max()
        stratum='mate_present' if anymate else ('below_-200' if cp.max() < -200 else 'above_200' if cp.max() > 200 else 'within_200')
        rec={'source_row_id':str(bank['source_row_ids'][i]),'group_id':str(bank['group_ids'][i]),
             'value_stratum':stratum,'best_effective_cp':float(cp.max()),
             'best_in_mate_band':bool(abs(cp.max())>SF_CP_CLAMP_CP),
             'legal_count':int(legal.sum()),'selected_size':int(selected.sum()),
             'stored_max_count':int(storedmax.sum()),'raw_cp_max_count':int(best.sum()),
             'selected_cp_spread':float(np.ptp(cp[selected])), 'stored_maxima_cp_spread':float(np.ptp(cp[storedmax])),
             'stored_maxima_q_spread':float(np.ptp(q[storedmax])),
             'q_spread':float(np.ptp(q)),'q_flat':bool(np.ptp(q)<1e-9),'q_exact_flat':bool(np.ptp(q)==0),
             'selected_SF_mass':float(normalize(sf[legal])[selected].sum()),'targets':{}}
        for label in ['SF','C20T05','BT4_T1','BT4_T05']:
            p=target[names.index(label),i]
            assert np.all(p[~legal] == 0)
            rec['targets'][label]=metrics(p[legal],selected,best)
        for temp in plan['temperatures_cp']:
            ideal=softmax_at_temp(cp,temp=temp)
            stored=shard_stored(ideal).astype(np.float64)
            rec['targets'][f'cp{temp}_ideal']=metrics(ideal,selected,best)
            rec['targets'][f'cp{temp}_stored']=metrics(stored,selected,best)
            rec['targets'][f'cp{temp}_stored']['mass_error']=float(abs(stored.sum()-1))
        records.append(rec)
    overall=summarize(records)
    strata={s:summarize([r for r in records if r['value_stratum']==s])
            for s in ['below_-200','within_200','above_200','mate_present']}
    intersections=Counter(f"{r['value_stratum']}|Csize={min(r['selected_size'],4)}(4means>3)|max={r['stored_max_count']}|qflat={r['q_flat']}" for r in records)
    closest={ref:min(plan['temperatures_cp'],key=lambda t:(abs(overall['targets'][f'cp{t}_stored']['entropy_nats']-overall['targets'][ref]['entropy_nats']),t)) for ref in ['SF','C20T05']}
    assert all(sha(p) == h for p,h in pins.items()), 'source/helper changed'
    result={'status':'COMPLETE_TRAINING_SAMPLE_ONLY','rows':128,'games':43,'pins':pins,
            'overall':overall,'SF_value_strata':strata,'intersections':dict(intersections),
            'descriptive_closest_mean_stored_entropy_temperature_cp':closest,
            'per_row':records,'elapsed_seconds':time.time()-started,
            'process_cpu_seconds':time.process_time(),'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'limits':plan['scope']+' No final temperature or gameplay choice; no full-corpus feasibility qualification.'}
    with (OUT/'readout.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds'],'sha256':sha(OUT/'readout.json')}))

if __name__ == '__main__':
    try:main()
    except BaseException as exc:
        with (OUT/'failed.json').open('x') as f:json.dump({'error':repr(exc)},f)
        raise
