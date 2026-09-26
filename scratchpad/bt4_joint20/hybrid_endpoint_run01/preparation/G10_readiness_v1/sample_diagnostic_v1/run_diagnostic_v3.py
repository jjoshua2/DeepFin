"""Frozen 128-row CPU diagnostic, not a training-corpus producer."""
from pathlib import Path
import collections
import hashlib
import itertools
import json
import os
import sys
import time

import numpy as np
import zarr
from numcodecs import blosc

W = Path('/tmp/deepfin-bt4-hybrid-tools')
sys.path.insert(0, str(W))
from scripts import derive_corpus_targets as derive
from scripts import gen_sf_rooted_corpus as gen
from scripts import sf_d9_rank_sidecar as ranks
from chess_anti_engine.eval.rvg_surgery import position_fingerprints

BASE = Path(__file__).resolve().parent
D = BASE / 'estimator_v3'
PLAN_SHA = 'd870714ef335e97a66fd24172754d48cad3ffd2cf3cb08d12dbfdec005804291'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    started = time.time()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    blosc.set_nthreads(2)
    D.mkdir(exist_ok=False)
    assert sha(BASE / 'preregistration.json') == PLAN_SHA
    plan = json.loads((BASE / 'preregistration.json').read_text())
    frozen = [json.loads(line) for line in (BASE / 'selected_raw_rows.jsonl').read_text().splitlines()]
    assert not (D / 'readout.json').exists()
    records, raw_records, join_records, matrices = [], [], [], []
    phases, errors = collections.Counter(), []
    for source in plan['sources']:
        sid, path, sidepath = source['source_id'], Path(source['source']), Path(source['sidecar'])
        assert sha(sidepath / '.zattrs') == source['sidecar_attrs_sha256']
        source_stat = path.stat()
        manifest = json.loads((path.parent / 'manifest.json').read_text())
        raw = [item['payload'] for item in frozen if item['source_id'] == sid]
        assert len(raw) == 64
        side = zarr.open_group(str(sidepath), mode='r')
        bt4 = np.asarray(side['bt4_policy'][:64])
        raw_input = np.asarray(side['input_key'][:64]); raw_key = np.asarray(side['source_key'][:64])
        gids = np.asarray(side['game_id'][:64]); plies = np.asarray(side['ply'][:64])
        options = derive.DeriveOptions(scheme=derive.Scheme(kind='uniform', depth=9), temp=.0005,
                                      cp_slope=.006, cp_draw_width=120., limit=64, seed=0,
                                      rows_per_shard=64, max_envelope_misses=0, floor=0., value_scheme='search')
        current = derive.TargetDeriver(options)
        samples, current_indices = [], []
        for index, row in enumerate(raw):
            raw_records.append({'source_id':sid,'source_shard':path.name,'physical_row':index,'payload':row})
            identity = {'source_id':sid,'source_shard':path.name,'physical_row':index,
                        'worker_id':row.get('worker_id'),'game_id':row.get('game_id'),'ply':row.get('ply'),
                        'input_key':row.get('input_key'),'search_key':row.get('search_key')}
            try:
                derive._check_row_identity(row, manifest['config_sha256'])
                board = derive.board_from_row(row)
                lines = ranks.d9_lines(row)
                legal = {m.uci() for m in board.legal_moves}
                assert len(lines) == len(legal) and {x[1] for x in lines} == legal
                assert len({x[0] for x in lines}) == len(lines)
                bank = derive.RowBank(row)
                latest = derive.apply_scheme(bank, options.scheme)
                phase0 = {str(x[1]):float(x[2]) for x in lines}
                cp0 = np.asarray([phase0[m] for m in latest.moves])
                cp1 = latest.effective_cp
                p0 = derive.softmax_at_temp(current.q_of(cp0), temp=.0005).astype(np.float32).astype(np.float16)
                p1 = derive.softmax_at_temp(current.q_of(cp1), temp=.0005).astype(np.float32).astype(np.float16)
                max0, max1 = p0 == p0.max(), p1 == p1.max()
                def close_set(cp, probs):
                    order = sorted(range(len(cp)), key=lambda i:(-cp[i], latest.moves[i]))[:3]
                    chosen = {i for i in order if cp.max()-cp[i] <= 20}
                    return chosen | set(np.flatnonzero(probs == probs.max()).tolist())
                set0, set1 = close_set(cp0,p0), close_set(cp1,p1)
                made = current.derive_row(row)
                assert made is not None,'current deriver dropped row (no result)'
                sample = made.sample
                assert gen.input_tensor_key(sample.x) == row['input_key'] == bytes(raw_input[index]).hex()
                finger = position_fingerprints(sample.x[None], input_history_encoding=derive.INPUT_HISTORY_ENCODING)[0]
                assert finger == bytes(raw_key[index])
                assert sample.game_id == int(gids[index]) == row['game_id']
                assert sample.ply_index == int(plies[index]) == row['ply']
                legal_mask=np.asarray(sample.legal_mask,dtype=bool)
                assert not np.any(bt4[index][~legal_mask]) and np.isfinite(bt4[index]).all()
                assert np.all(bt4[index]>=0) and abs(float(bt4[index].sum(dtype=np.float64))-1)<2e-6
                # Preserve current latest-phase searched-value behavior in this diagnostic writer.
                derive.write_value_target(sample, made.facts.q_wdl)
                samples.append(sample); current_indices.append(index)
                phase_count=len(row['phases']);phases[(sid,phase_count)]+=1
                record=identity|{'legal_moves':len(legal),'phase0_d9_complete_all_legal':True,
                                'phase_count':phase_count,'latest_move_phase_counts':dict(collections.Counter(latest.phase_by_move)),
                                'cp_changed_moves':int(np.count_nonzero(cp0!=cp1)),
                                'max_abs_cp_delta':float(np.abs(cp1-cp0).max()),
                                'argmax_changed':latest.moves[int(np.argmax(cp0))]!=latest.moves[int(np.argmax(cp1))],
                                'stored_max_set_changed':not np.array_equal(max0,max1),
                                'consistent_close_set_changed':set0!=set1,
                                'phase0_rank1_is_latest_stored_max':bool(max1[int(np.argmax(cp0))]),
                                'stored_policy_l1':float(np.abs(p0.astype(float)/p0.sum(dtype=float)-p1.astype(float)/p1.sum(dtype=float)).sum()),
                                'phase0_best_cp':float(cp0.max()),'latest_best_cp':float(cp1.max()),
                                'searched_wdl_max_abs_delta':float(np.abs(current.wdl_of(float(cp0.max()))-made.facts.q_wdl).max()),
                                'current_deriver_source_and_input_keys_match_raw_bt4':True}
                records.append(record)
                matrices.append({'x':sample.x,'bt4':bt4[index],'legal':sample.legal_mask})
            except Exception as exc:
                errors.append(identity|{'error':repr(exc)})
        if samples:
            output=D/(sid+'_diagnostic_current_writer');output.mkdir()
            commit=derive.CommitIdentity(corpus_complete=False, corpus_run_finished_claim=False,
                                        corpus_shards_adopted=1,corpus_rows_claimed=64,corpus_record_row_schema=3)
            derive._flush(output,0,samples,options,np.random.default_rng(0),manifest['config_sha256'],
                          schemas=[3]*len(samples),identity=commit)
            group=zarr.open_group(str(output/'shard_000000.zarr'),mode='r')
            order=np.random.default_rng(0).permutation(len(samples))
            for dest, original in enumerate(order):
                rawindex=current_indices[int(original)];sample=samples[int(original)]
                x=np.asarray(group['x'][dest]);legal=np.asarray(group['legal_mask'][dest]);key=gen.input_tensor_key(x)
                raw_key_matches_stored = key == bytes(raw_input[rawindex]).hex()
                expected_stored = sample.x.astype(x.dtype)
                assert key == gen.input_tensor_key(expected_stored)
                assert position_fingerprints(x[None],input_history_encoding=derive.INPUT_HISTORY_ENCODING)[0]==bytes(raw_key[rawindex])
                assert np.array_equal(x,expected_stored) and np.array_equal(legal,sample.legal_mask)
                assert int(group['game_id'][dest])==int(gids[rawindex]) and int(group['ply_index'][dest])==int(plies[rawindex])
                assert np.array_equal(np.asarray(group['policy_target'][dest]),sample.policy_target.astype(np.float16))
                assert np.array_equal(np.asarray(group['search_wdl'][dest]),np.asarray(sample.search_wdl).astype(np.float16))
                join_records.append({'source_id':sid,'source_shard':path.name,'raw_row_index':rawindex,
                                     'diagnostic_derived_shard':'shard_000000.zarr','derived_row':dest,
                                     'game_id':int(gids[rawindex]),'ply':int(plies[rawindex]),'stored_input_key':key,
                                     'raw_input_key':bytes(raw_input[rawindex]).hex(),
                                     'raw_input_key_survives_storage_exactly':raw_key_matches_stored,
                                     'stored_x_equals_quantized_original':True,
                                     'stored_x_max_abs_quantization_error':float(np.abs(x.astype(np.float32)-sample.x).max()),
                                     'source_key':bytes(raw_key[rawindex]).hex(),'bt4_policy_row_sha256':hashlib.sha256(bt4[rawindex].tobytes()).hexdigest()})
        after=path.stat()
        assert (after.st_ino,after.st_size,after.st_mtime_ns)==(source_stat.st_ino,source_stat.st_size,source_stat.st_mtime_ns)
        assert sha(sidepath/'.zattrs')==source['sidecar_attrs_sha256']
    for name, rows in [('errors.jsonl',errors),('row_readout.jsonl',records),('row_join.jsonl',join_records)]:
        with (D/name).open('x') as f:
            for row in rows:f.write(json.dumps(row,sort_keys=True)+'\n')
    np.savez_compressed(D/'selected_arrays.npz',x=np.stack([a['x'] for a in matrices]),
                        bt4=np.stack([a['bt4'] for a in matrices]),legal=np.stack([a['legal'] for a in matrices]))
    bare=collections.Counter((r['game_id'],r['ply']) for r in records)
    result={'schema':1,'status':'PASS_BOUNDED_OBSERVATION_AND_JOIN' if not errors else 'PARTIAL_WITH_ROW_ERRORS',
            'preregistration_sha256':PLAN_SHA,'frozen_raw_bank_sha256':sha(BASE/'selected_raw_rows.jsonl'),'script_sha256':sha(__file__),'elapsed_seconds':time.time()-started,
            'sampled_rows':len(raw_records),'valid_complete_d9_derived_join_rows':len(records),'writer_roundtrip_rows':len(join_records),
            'source_qualified_games':len({(r['source_id'],r['game_id']) for r in records}),
            'phase_count_strata':{str(k):v for k,v in phases.items()},'rows_with_changed_cp':sum(r['cp_changed_moves']>0 for r in records),
            'rows_with_argmax_change':sum(r['argmax_changed'] for r in records),
            'rows_with_stored_max_set_change':sum(r['stored_max_set_changed'] for r in records),
            'rows_with_consistent_close_set_change':sum(r['consistent_close_set_changed'] for r in records),
            'rows_where_phase0_rank1_not_latest_stored_max':sum(not r['phase0_rank1_is_latest_stored_max'] for r in records),
            'mean_stored_policy_l1':float(np.mean([r['stored_policy_l1'] for r in records])),
            'rows_with_changed_searched_wdl':sum(r['searched_wdl_max_abs_delta']>0 for r in records),
            'max_searched_wdl_abs_delta':max(r['searched_wdl_max_abs_delta'] for r in records),
            'raw_input_key_changes_after_float16_storage_rows':sum(not r['raw_input_key_survives_storage_exactly'] for r in join_records),
            'max_input_storage_abs_quantization_error':max(r['stored_x_max_abs_quantization_error'] for r in join_records),
            'bare_game_ply_keys_colliding_across_sources':sum(v>1 for v in bare.values()),'errors':errors,
            'limitations':['Training-only deterministic first64 rows of two closed shards; limited games and phase coverage, no prevalence/heldout/strength claim.',
                           'Current writer exercised on64-row diagnostic shards, not production8192-row sequence/parallel scheduling.',
                           'Current latest-phase search_wdl was preserved; phase0 value differences are diagnostic only, not a selected transfer value target.',
                           'No full corpus qualification, inference, GPU, generator or labeler mutation.'],
            'artifacts':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in D.iterdir() if p.is_file()},
            'implementation_sha256':{str(p.relative_to(W)):sha(p) for p in [W/'scripts/derive_corpus_targets.py',W/'scripts/gen_sf_rooted_corpus.py',W/'scripts/sf_d9_rank_sidecar.py',W/'scripts/bt4_raw_corpus_sidecar.py']}}
    with (D/'readout.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifacts','implementation_sha256','errors')},indent=2))

if __name__=='__main__':main()
