"""Completed-bank review; standalone arithmetic, no arena reader or model imports."""
import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
import numpy as np

root = Path('/home/josh/projects/chess/scratchpad/bt4_joint20/hybrid_endpoint_run01')
state = root/'H20'
cell = state/'C20T05.s400'
out = cell/'independent_calculation'
pins = {}
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    path = Path(path)
    pins[str(path)] = sha(path)
    return json.loads(path.read_text())
def pin(ref):
    actual = sha(ref['path'])
    assert actual == ref['sha256'], ref
    pins[ref['path']] = actual

def readbank(name, n, sims):
    d = state/name
    complete = load(d/'complete.json'); process = load(d/'process.json')
    manifest = load(d/'manifest.json'); automatic = load(d/'readout.json')
    result = load(d/'arena.results.jsonl')
    assert complete['complete'] is True and complete['process_complete'] is True
    assert process['process_complete'] is True and process['exit_code'] == complete['exit_code'] == 0
    assert not (d/'failed.json').exists()
    assert complete['command'] == complete['arena_cmdline'] == process['command']
    assert complete['command'][1:] == result['argv']
    assert '--no-rolling' in complete['command'] and '--sprt' not in complete['command']
    assert complete['ended_unix'] > complete['started_unix']
    assert abs(complete['ended_unix']-complete['started_unix']-complete['gpu_seconds']) < .01
    assert 0 < complete['gpu_seconds'] <= complete['hard_seconds'] == 5400
    for k in ('models','runtime','cwd','live_config','book','launcher_sha256'):
        assert complete[k] == process[k]
    for k in ('candidate','reference'):
        assert complete['models'][k] == manifest[k]
    for k in ('preregistration','reader','runtime_manifest','candidate_training'):
        pin(manifest[k])
    assert complete['reader_sha256'] == manifest['reader']['sha256']
    assert complete['launcher_sha256'] == manifest['launcher_sha256']
    assert complete['book'] == manifest['book']
    path = d/'arena.games.jsonl'; pins[str(path)] = sha(path)
    assert pins[str(path)] == complete['games_sha256'] == automatic['sha256']
    assert sha(d/'readout.json') == complete['readout_sha256']
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows)==2*n+1 and rows[0]['kind']=='header'
    header=rows[0]; settings=header['settings']; games=rows[1:]
    assert all(row['kind']=='game' for row in games)
    assert settings == automatic['settings']
    assert settings['games']==2*n and settings['sims_candidate']==settings['sims_reference']==sims
    assert settings['candidate']==manifest['candidate']['path'] and settings['reference']==manifest['reference']['path']
    assert settings['seed']==42 and settings['temperature']==.1 and settings['max_plies']==300
    assert settings['opening_plies']==16 and settings['openings']==manifest['book']['path']
    assert settings['mode']=='matched_sims' and settings['gumbel_add_noise'] is True
    assert settings['search_candidate']==settings['search_reference']
    assert settings['search_candidate']['gumbel']['policy_temp']==1.
    assert result['games']==2*n and result['pairs']==n and result['games_requested']==2*n
    assert result['game_log_agrees'] is True and result['truncated'] is False
    assert result['resumed_pairs']==result['resumed_orphan_pairs']==0 and 'sprt' not in result
    assert result['compile']=='on' and result['eval_hoist']=='4096'
    assert result['mixed_compile'] is False and result['mixed_eval_hoist'] is False
    assert result['max_concurrent_games']==128 and result['eval_max_batch']==4096
    assert result['git_sha']=='7ec261509fb7345cf1ca0ad73809193fc2749bb1'
    assert result['config_name']=='pbt2_small.yaml' and result['config_authoritative'] is True
    assert result['game_log_fingerprint']==header['fingerprint']
    for k in ('search_candidate','search_reference','candidate','reference','sims_candidate','sims_reference','seed','temperature','max_plies'):
        assert result[k]==settings[k]
    indexed={}; wdl=Counter(); endings=Counter()
    for row in games:
        i,h=row['pair_id'],row['half']
        assert type(i) is int and 0<=i<n and type(h) is int and h in (0,1)
        assert (i,h) not in indexed and row['opening_index']==i
        assert row['a_is_white'] is (h==0)
        assert row['compile']=='on' and row['eval_hoist']=='4096' and row['seed']==42
        assert row['loop']=='chunked' and row['opening_fen']==row['start_fen']
        assert row['termination'] in ('rules','max_plies')
        if row['result'] in ('1/2-1/2','*'):
            score=.5
            if row['result']=='*': assert row['termination']=='max_plies'
        else:
            assert row['result'] in ('1-0','0-1')
            score=float((row['result']=='1-0')==row['a_is_white'])
        assert score==row['score_candidate']
        indexed[i,h]=(score,row['start_fen'])
        wdl[{0.:'losses',.5:'draws',1.:'wins'}[score]]+=1
        endings[row['termination']]+=1
    pair_scores=[]; fens=[]
    for i in range(n):
        assert indexed[i,0][1]==indexed[i,1][1]
        fens.append(indexed[i,0][1]); pair_scores.append((indexed[i,0][0]+indexed[i,1][0])/2)
    assert len(set(fens))==n
    a=np.array(pair_scores,dtype=np.float64)
    mean=float(a.mean()); se=float(a.std(ddof=1)/math.sqrt(n)); ci=[mean-1.96*se,mean+1.96*se]
    elo=lambda p:400*math.log10(p/(1-p))
    pent=Counter({k:pair_scores.count(v) for v,k in zip((0.,.25,.5,.75,1.),('LL','LD_DL','DD_WL','WD_DW','WW'))})
    calc={**wdl,'games':2*n,'pairs':n,'score':mean,'score_se':se,'score_ci95':ci,'elo':elo(mean),'elo_ci95':list(map(elo,ci)),'pentanomial':dict(pent),'terminations':dict(endings)}
    for k in ('score','elo','score_ci95','elo_ci95'):
        assert np.allclose(calc[k],automatic['result'][k],rtol=0,atol=1e-12), k
    assert calc['pentanomial']==automatic['result']['pentanomial']==result['pentanomial']
    assert automatic['match_complete'] is True and automatic['raw_game_rows']==2*n
    assert automatic['superseded_orphan_rows']==0 and automatic['execution']==['on','4096']
    return {'calc':calc,'pairs':a,'fens':fens,'complete':complete,'manifest':manifest,'settings':settings}

launch=load(root/'H20.launch_manifest.json'); pin(launch['preregistration'])
assert launch['comparisons']==[['C20T05',100,1000],['G20T05',100,1000],['C20T05',400,500]]
training=load(state/'training.complete.json'); package=load(state/'complete.json')
assert training['complete'] is True and training['historical_valid_control'] is False
assert training['canonical_plan_sha256']=='dc687fc333295dee565d19bb4f20da5aa95479dba3aacc5499c22a4004acc64f'
pin(training['schedule'])
assert package['complete'] is True and package['profile']=='H20'
assert package['training_receipt_sha256']==sha(state/'training.complete.json')
for x in package['arena_receipts']:pin(x)
terminal=load(cell/'independent_terminal_observation.json')
assert not any(terminal['pids'].values())
prior_c=load(state/'C20T05.s100/independent_review.json')
prior_g=load(state/'G20T05.s100/independent_review.json')
for review in (prior_c,prior_g):
    for path,digest in review['input_hashes'].items():
        assert sha(path)==digest; pins[path]=digest
c100=readbank('C20T05.s100',500,100)
c400=readbank('C20T05.s400',250,400)
assert c400['fens']==c100['fens'][:250]
assert c400['complete']['models']==c100['complete']['models']
assert training['checkpoint']==c400['complete']['models']['candidate']
for k in ('runtime','cwd','live_config','book','qualified_search_readout_sha256'):
    assert c400['complete'][k]==c100['complete'][k]
for k,v in c100['settings'].items():
    if k not in ('games','sims_candidate','sims_reference'):assert c400['settings'][k]==v,k
for x in c400['manifest']['opening_anchor'].values():pin(x)

low=c100['pairs'][:250]; high=c400['pairs']; delta=high-low
rng=np.random.Generator(np.random.PCG64(20260903))
# One registered 10,000-replicate bootstrap, resampling aligned opening pairs.
boot=np.empty(10000,dtype=np.float64)
for start in range(0,10000,1000):
    indices=rng.integers(0,250,size=(1000,250))
    boot[start:start+1000]=delta[indices].mean(axis=1)
interval=np.quantile(boot,[.025,.975],method='linear').tolist()
interaction={'exploratory':True,'estimand':'mean aligned opening-pair candidate score at400 minus100 simulations; pair score is mean of both color games','pairs':250,'score100_first250':float(low.mean()),'score400':float(high.mean()),'delta_score':float(delta.mean()),'delta_percentage_points':float(delta.mean()*100),'bootstrap_ci95_score':interval,'bootstrap_ci95_percentage_points':[x*100 for x in interval],'bootstrap':{'generator':'PCG64','seed':20260903,'replicates':10000,'percentiles':[2.5,97.5],'quantile_method':'linear','numpy':np.__version__},'interpretation':'No resolved change in relative score across these two budgets' if interval[0]<=0<=interval[1] else 'Positive relative-score interaction' if interval[0]>0 else 'Negative relative-score interaction'}
with (out/'aligned_pairs.csv').open('x',newline='') as f:
    w=csv.writer(f);w.writerow(['pair_id','opening_fen','score100','score400','delta_score'])
    for i in range(250):w.writerow([i,c400['fens'][i],low[i],high[i],delta[i]])
with (out/'bootstrap_replicates.npy').open('xb') as f:np.save(f,boot,allow_pickle=False)
charges=[load(Path(x['path']))['gpu_seconds'] for x in package['arena_receipts']]
total=training['training_charge_seconds']+sum(charges)
assert abs(total-package['gpu_seconds'])<1e-8 and total<=launch['total_seconds']==32400
c=c400['calc']; primary=c100['calc']; g=prior_g['computed']
assert primary['elo']>=15 and primary['elo_ci95'][0]>0 and g['elo_ci95'][0]>0
receipt={'schema':1,'status':'PASS_COMPLETE_REGISTERED_H20_PACKAGE','reviewed_unix':time.time(),'findings':[],'computed_c400':c,'computed_c100':primary,'g100_reused_independently_reviewed_result':g,'aligned_search_interaction':interaction,'registered_package_verdict':{'C100':'PROMISING: point>=15 Elo and lower bound>0','G100':'FAVORS_H20: wholly positive interval','C400':'Positive higher-search probe; not a separately registered primary-screen classification','package':'H20 promising versusC at100; specified selected-set construction favored versusG at100; H20 also wins registered400 probe. No resolved increasing search advantage.'},'protocol_checks':['Exactly500 C400 games/250 complete unique pairs, no duplicates/orphans/replay; all500 C100 pairs independently rescored','C400 endpoints and both colors match exactly first250 C100 canonical opening IDs','Full settings identical between budgets except requestedgames/sims; noSPRT or rolling execution','Checkpoint paths/SHA lineage, book, actualruntime/native paths, training schedule receipt, preregistration and opening anchor pins agree','Terminal process/complete/package receipts and actual host PID absence checked; no failed receipt; actual stage charge closed','Result/automatic readout and independent unrounded pair-sample-variance95percent normal interval agree'], 'charges':{'training_seconds':training['training_charge_seconds'],'arena_seconds':charges,'c400_seconds':c400['complete']['gpu_seconds'],'package_gpu_seconds':total,'package_gpu_hours':total/3600,'hard_package_seconds':launch['total_seconds']},'historical_valid_control':False,'historical_validity_problems':training['historical_validity_problems'],'limits':['Same seed-zero trajectories and reused development openings; intervals do not include training-seed variation or adaptive selection uncertainty.','400-minus100 contrast is exploratory with registered aligned250-pair PCG64 bootstrap; no Elo subtraction, no full scaling curve claim.','Independent raw arithmetic and metadata checks; original launcher checkpoint payload hashes and earlier training qualification reused rather than reread.','Reviewer authored portions of historical launcher plumbing; this is independent completed numerical/accounting review, not a fresh review of all inherited implementation.','No games, model inference, GPU access, training, corpus scans or runtime mutation.'], 'input_hashes':pins,'calculation_script_sha256':sha(__file__),'artifacts':{str(out/x):sha(out/x) for x in ('aligned_pairs.csv','bootstrap_replicates.npy')}}
for path,digest in pins.items():assert sha(path)==digest
with (cell/'independent_review.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({'status':receipt['status'],'c400':c,'interaction':interaction,'charges':receipt['charges'],'receipt_sha256':sha(cell/'independent_review.json')}))
